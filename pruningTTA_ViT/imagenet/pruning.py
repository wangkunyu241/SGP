import torch
import torch.nn as nn
import torch.jit
import math
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from timm.data import resolve_model_data_config, create_transform

def is_frozen_layer(name):
    filter_keywords = ['blocks.11', 'norm.']
    
    if name == 'norm':
        return True
    
    if any(k in name for k in filter_keywords):
        return True
        
    return False

def collect_params(model):
    params = []
    params_names = []
    
    for name, module in model.named_modules():
        if is_frozen_layer(name):
            continue

        if isinstance(module, nn.LayerNorm):
            for param_name, param in module.named_parameters():
                if param.requires_grad and param_name in ['weight', 'bias']:
                    params.append(param)
                    full_name = f"{name}.{param_name}"
                    params_names.append(full_name)
                    
    return params, params_names

def configure_model(model):
    model.eval() 
    model.requires_grad_(False)
    
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            # 使用统一过滤逻辑
            if is_frozen_layer(name):
                continue
            m.requires_grad_(True)
            
    return model

class Pruning(nn.Module):
    def __init__(self, model, optimizer, lamda_std, lamda_align, e_margin, rate, lamda_sparse, threshold, lr_mask, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.lamda_std = lamda_std
        self.lamda_align = lamda_align
        self.e_margin = e_margin
        self.rate = rate
        self.lamda_sparse = lamda_sparse
        self.threshold = threshold
        self.lr_mask = lr_mask

        self.current_feature = None 
        self.hook_handle = self.model.norm.register_forward_hook(self._save_feature_hook)
        
        self.num_prune_layers = 10
        self.num_heads = 12
        
        self.head_masks = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_heads, device="cuda")) for _ in range(self.num_prune_layers)
        ])
        
        self.register_buffer('head_active', torch.ones(self.num_prune_layers, self.num_heads, device="cuda"))
        
        self.current_head_stats_batch = [None] * self.num_prune_layers
        
        self._register_pruning_hooks()
        
        self.optimizer.add_param_group({
            'params': self.head_masks.parameters(),
            'lr': self.lr_mask
        })

        self.stats_file = "./source_stats/imagenet_stats_pruning.pt"
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        self._load_or_compute_stats()
        
    def _save_feature_hook(self, module, input, output):
        self.current_feature = output[:, 0]

    def _register_pruning_hooks(self):
        for i in range(self.num_prune_layers):
            proj_module = self.model.blocks[i].attn.proj
            proj_module.register_forward_pre_hook(self._make_proj_pre_hook(i))

    def _make_proj_pre_hook(self, layer_idx):
        def hook(module, args):
            x = args[0]
            B, N, C = x.shape
            head_dim = C // self.num_heads
            
            x_reshaped = x.view(B, N, self.num_heads, head_dim)
            
            stat = x_reshaped.mean(dim=(1, 3)) 
            self.current_head_stats_batch[layer_idx] = stat
            
            with torch.no_grad():
                dropped = self.head_masks[layer_idx] < self.threshold
                self.head_active[layer_idx][dropped] = 0.0
                
            current_active = self.head_active[layer_idx].clone().detach()
            effective_mask = self.head_masks[layer_idx] * current_active
            
            x_masked = x_reshaped * effective_mask.view(1, 1, self.num_heads, 1)
            
            return (x_masked.view(B, N, C),)
        return hook

    def _load_or_compute_stats(self, num_samples=10000):
        if os.path.exists(self.stats_file):
            print(f"Loading source stats from {self.stats_file}")
            stats = torch.load(self.stats_file, map_location="cuda")
            self.source_mean = stats['mu']
            self.source_std = stats['std']
            self.source_head_stats = stats['head_stats']
            return

        print("Computing source statistics (Alignment & Head Pruning)...")
        self.model.eval()
        
        data_config = resolve_model_data_config(self.model)
        transform = create_transform(**data_config, is_training=False)
        
        train_dir = 'your_imagenet_train_dir'
        dataset = ImageFolder(train_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        features_list = []
        count = 0
        entropy_threshold = math.log(1000)/2 - 1

        running_head_sum = torch.zeros(self.num_prune_layers, self.num_heads, device='cuda')
        
        with torch.no_grad():
            for images, _ in tqdm(loader):
                images = images.cuda()
                outputs = self.model(images)
                
                entropys = softmax_entropy_align(outputs)
                selected_indices = torch.where(entropys < entropy_threshold)[0]
                
                if len(selected_indices) == 0: continue
                
                selected_feat = self.current_feature[selected_indices]
                features_list.append(selected_feat.cpu())

                for i in range(self.num_prune_layers):
                    running_head_sum[i] += self.current_head_stats_batch[i][selected_indices].sum(dim=0)

                count += len(selected_indices)
                if count >= num_samples: break
        
        full_feats = torch.cat(features_list, dim=0)[:num_samples]
        std, mu = torch.std_mean(full_feats, dim=0)
        
        self.source_mean = mu.cuda()
        self.source_std = std.cuda()

        self.source_head_stats = running_head_sum / count

        torch.save({
            'mu': self.source_mean, 
            'std': self.source_std,
            'head_stats': self.source_head_stats
        }, self.stats_file)
        torch.cuda.empty_cache()

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        outputs = self.model(x)

        curr_feat = self.current_feature 
        batch_std, batch_mean = torch.std_mean(curr_feat, dim=0)

        src_mean = self.source_mean.detach()
        src_std = self.source_std.detach()

        std_loss = torch.norm(batch_std - src_std, p=2)
        mean_loss = torch.norm(batch_mean - src_mean, p=2)
        loss_align = mean_loss + self.lamda_std * std_loss
        
        loss_ent = entropy_minmization(outputs, self.e_margin)

        loss_prune = torch.tensor(0.0, device=x.device)
        
        pruned_heads = (self.head_active == 0).sum().item()

        total_model_heads = 120
        prune_ratio = pruned_heads / total_model_heads
        
        if prune_ratio < self.rate:
            target_head_stats = torch.stack([
                self.current_head_stats_batch[i].mean(dim=0)
                for i in range(self.num_prune_layers)
            ])
            
            diff = torch.abs(target_head_stats.detach() - self.source_head_stats.detach())

            active_bool_mask = (self.head_active == 1)
            
            active_diffs = diff[active_bool_mask]
            
            diff_min = active_diffs.min()
            diff_max = active_diffs.max()
            
            weight = (diff - diff_min) / (diff_max - diff_min + 1e-8) 
            
            mask_tensor = torch.stack(list(self.head_masks))
            active_clone = self.head_active.clone().detach()
            
            loss_prune = torch.sum(weight * torch.abs(mask_tensor * active_clone))

        loss = loss_align + self.lamda_align * loss_ent + self.lamda_sparse * loss_prune

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return outputs

@torch.jit.script
def softmax_entropy_align(x: torch.Tensor) -> torch.Tensor:
    temprature = 1
    x = x/ temprature
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def entropy_minmization(outputs, e_margin=0.4):
    probs = outputs.softmax(1)
    log_probs = outputs.log_softmax(1)
    entropys = -(probs * log_probs).sum(1)
    
    filter_ids = torch.where(entropys < e_margin)[0]
    filtered_entropys = entropys[filter_ids]
    
    if filtered_entropys.shape[0] == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    
    return filtered_entropys.mean(0)