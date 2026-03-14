import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
from methods.base import TTAMethod
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

def resnext_patched_forward(self, x):
    residual = x
    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)
    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)
    if self.downsample is not None:
        residual = self.downsample(x)
    
    out_before_relu = residual + bottleneck
    self.latten_features = out_before_relu

    return F.relu(out_before_relu, inplace=True)

def resnet_patched_forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual
    
    self.latten_features = out 
    
    out = self.relu(out)

    return out

class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sum_sq = 0

    def update(self, x):
        x = x.detach().cpu()
        self.n += x.shape[0]
        self.sum += x.sum(dim=0)
        self.sum_sq += (x ** 2).sum(dim=0)

    def mean(self):
        return self.sum / self.n

    def var(self):
        return (self.sum_sq / self.n) - (self.mean() ** 2)

def entropy_minmization(outputs, e_margin=0.4):
    entropys = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    filter_ids_1 = torch.where(entropys < e_margin)
    entropys = entropys[filter_ids_1]
    if entropys.shape[0] == 0:
        return torch.tensor(0.0).to(outputs.device)
    ent = entropys.mean(0)
    return ent

class Pruning(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
     
        # --- 路径配置 ---
        self.stats_dir = "./source_stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.stat_file = os.path.join(
            self.stats_dir, 
            f"{self.cfg.CORRUPTION.DATASET.split('_')[0]}_stats.pt"
        )
        
        self.current_alignment_feats = {}  
        self.current_bn_feats = {}         
        
        self.patched_blocks = {} 
        
        self.pruning_masks = {}            

        self.setup_hooks()
        self.source_stats = self.collect_source_stat()
        self.init_masks()

    def get_pruning_rate(self):
        total_channels = 0
        pruned_channels = 0
        
        with torch.no_grad():
            for name, mask in self.pruning_masks.items():
                total_channels += mask.numel()
                pruned_channels += (mask == 0).sum().item()
        
        if total_channels == 0:
            return 0.0
            
        return pruned_channels / total_channels
   
    def init_masks(self):
        dataset_name = self.cfg.CORRUPTION.DATASET.split("_")[0]
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if dataset_name == "cifar10" and ("block3" in name or name == "bn1"):
                    continue 
                if dataset_name == "cifar100" and "stage_3" in name:
                    continue
                if dataset_name == "imagenet" and ("layer4" in name or "layer3.5" in name):
                    continue
                self.pruning_masks[name] = torch.ones_like(module.weight.data).to(self.cfg.DEVICE)
                

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        
        dataset_name = self.cfg.CORRUPTION.DATASET.split("_")[0]
        
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                
                if dataset_name == "cifar10" and ("block3" in name or name == "bn1"):
                    m.requires_grad_(False)
                if dataset_name == "cifar100" and "stage_3" in name:
                    m.requires_grad_(False)
                if dataset_name == "imagenet" and ("layer4" in name or "layer3.5" in name):
                    m.requires_grad_(False)

    def get_intermediate_hook(self, name):
        def hook(module, input, output):
            self.current_alignment_feats[name] = output
        return hook
    
    def hook_fn_alignment(self, module, input, output):
        self.current_alignment_feats['final'] = input[0]
    
    def get_bn_input_hook(self, name):
        def hook(module, input, output):
            x = input[0]
            if x.dim() == 4:
                feats = x.mean(dim=(2, 3))
            else:
                raise ValueError("Wrong BN Feature Size")
            self.current_bn_feats[name] = feats
        return hook

    def setup_hooks(self):
        dataset_name = self.cfg.CORRUPTION.DATASET.split("_")[0]
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if dataset_name == "cifar10" and ("block3" in name or name == "bn1"):
                    continue 
                if dataset_name == "cifar100" and "stage_3" in name:
                    continue
                if dataset_name == "imagenet" and ("layer4" in name or "layer3.5" in name):
                    continue
                module.register_forward_hook(self.get_bn_input_hook(name))

        if dataset_name == "cifar10":
            if hasattr(self.model, 'relu'):
                self.model.relu.register_forward_hook(self.hook_fn_alignment)

        elif dataset_name == "cifar100":
            s3_block = self.model.stage_3[2]
            s3_block.forward = types.MethodType(resnext_patched_forward, s3_block)
            self.patched_blocks['final'] = s3_block

        elif dataset_name == "imagenet":
            base_model = self.model.model if hasattr(self.model, 'model') else self.model
            l4_block = base_model.layer4[-1] 
            l4_block.forward = types.MethodType(resnet_patched_forward, l4_block)
            self.patched_blocks['final'] = l4_block

    def sync_patched_features(self):
        for name, block in self.patched_blocks.items():
            if hasattr(block, 'latten_features'):
                self.current_alignment_feats[name] = block.latten_features

    def collect_source_stat(self):
        if os.path.exists(self.stat_file):
            logger.info(f"Loading cached source stats from {self.stat_file}")
            return torch.load(self.stat_file, map_location=self.cfg.DEVICE)

        logger.info("Starting source stats calculation (Multi-layer Patching)...")
        dataset_name = self.cfg.CORRUPTION.DATASET.split("_")[0]
        loader = self.get_source_loader(dataset_name)
        
        align_runners = {} 
        bn_runners = {}
        
        max_samples = 50000
        limit_batches = max_samples // self.cfg.TEST.BATCH_SIZE
        total_batches = min(len(loader), limit_batches)
        
        self.model.eval()
        with torch.no_grad():
            for i, (imgs, _) in tqdm(enumerate(loader), total=total_batches):
                if i >= limit_batches: break

                imgs = imgs.to(self.cfg.DEVICE)
                _ = self.model(imgs)
                
                self.sync_patched_features()
                
                for layer_name, feats in self.current_alignment_feats.items():
                    if layer_name not in align_runners:
                        align_runners[layer_name] = RunningStats()
                    
                    if feats.dim() == 4:
                        feats_pooled = feats.mean(dim=(2, 3))
                        align_runners[layer_name].update(feats_pooled)
                
                # 3. 统计 BN 层
                for name, feats in self.current_bn_feats.items():
                    if name not in bn_runners:
                        bn_runners[name] = RunningStats()
                    bn_runners[name].update(feats)

        # 保存
        final_stats = {
            "alignment": {}, 
            "all_bns": {}
        }
        for layer_name, runner in align_runners.items():
            final_stats["alignment"][layer_name] = {
                "mu": runner.mean().to(self.cfg.DEVICE),
                "std": torch.sqrt(runner.var().clamp(min=1e-8)).to(self.cfg.DEVICE)
            }
        for name, runner in bn_runners.items():
            final_stats["all_bns"][name] = runner.mean().to(self.cfg.DEVICE)
            
        torch.save(final_stats, self.stat_file)
        return final_stats
    
    def get_source_loader(self, dataset_name):
        logger.info(f"Setting up loader for clean {dataset_name}...")
        
        if dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            dataset = torchvision.datasets.CIFAR10(
                root="your_cifar10_dir", train=True, download=False, transform=transform)
            
        elif dataset_name == "cifar100":
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            dataset = torchvision.datasets.CIFAR100(
                root="your_cifar100_dir", train=True, download=False, transform=transform)
            
        elif dataset_name == "imagenet":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            dataset = torchvision.datasets.ImageFolder(
                root="your_imagenet_train_dir",
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=min(self.cfg.TEST.NUM_WORKERS, os.cpu_count()),
            drop_last=False
        )
        return loader
    
    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0].to(self.cfg.DEVICE)
        
        current_pruning_rate = self.get_pruning_rate()
        target_pruning_rate = self.cfg.PRUNING.RATE

        self.apply_pruning_mask()
        
        outputs = self.model(imgs_test)
        
        self.sync_patched_features()

        loss_ent = entropy_minmization(outputs, self.cfg.PRUNING.LAMDA_MARGIN)

        loss_align = 0.0
        
        for layer_name, curr_feats in self.current_alignment_feats.items():
            curr_pooled = curr_feats.mean(dim=(2, 3))
            curr_mu = curr_pooled.mean(dim=0)
            curr_std = torch.sqrt(curr_pooled.var(dim=0, unbiased=False) + 1e-8)
            
            src_mu = self.source_stats["alignment"][layer_name]["mu"].detach()
            src_std = self.source_stats["alignment"][layer_name]["std"].detach()
            
            layer_loss = F.mse_loss(curr_mu, src_mu) + self.cfg.PRUNING.LAMDA_MU_STD * F.mse_loss(curr_std, src_std)
            
            loss_align += layer_loss

        if current_pruning_rate < target_pruning_rate:
            loss_sparse = self.calculate_sparse_loss()
            term_sparse = self.cfg.PRUNING.LAMDA_SPARSE * loss_sparse
        else:
            loss_sparse = torch.tensor(0.0).to(self.cfg.DEVICE)
            term_sparse = 0.0

        # 总损失权重
        loss = loss_ent + self.cfg.PRUNING.LAMDA_ALIGN * loss_align + term_sparse

        # 4. 反向传播
        loss.backward()
        
        self.mask_gradients()
        
        self.optimizer.step()
        
        if current_pruning_rate < target_pruning_rate:
            self.prune_parameters(threshold=self.cfg.PRUNING.THRESHOLD)
        
        self.optimizer.zero_grad()
        
        return outputs

    def calculate_sparse_loss(self):
        sensitivity_list = []
        bn_modules = []
        active_sens_list = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in self.current_bn_feats:
                curr_feats = self.current_bn_feats[name]
                curr_mean = curr_feats.mean(dim=0)
                
                if name in self.source_stats["all_bns"]:
                    src_mean = self.source_stats["all_bns"][name].detach()
                else:
                    raise ValueError("Wrong BN Layer Name")
                
                diff = torch.abs(curr_mean - src_mean)
                sensitivity_list.append(diff)
                bn_modules.append(module)

                mask = self.pruning_masks.get(name, torch.ones_like(diff))
                
                active_diff = diff[mask == 1]
                if active_diff.numel() > 0:
                    active_sens_list.append(active_diff)
        
        if not sensitivity_list:
            raise ValueError("No Sensitivity List, No BN?")

        all_active_sens = torch.cat(active_sens_list)
        s_min = all_active_sens.min()
        s_max = all_active_sens.max()
        
        denom = s_max - s_min + 1e-6
        
        total_sparse_loss = 0
        
        for i, module in enumerate(bn_modules):
            sens = sensitivity_list[i]

            w = (sens - s_min) / denom
            
            layer_loss = (w * torch.abs(module.weight)).sum()
            total_sparse_loss += layer_loss
            
        return total_sparse_loss

    def apply_pruning_mask(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.pruning_masks:
                    mask = self.pruning_masks[name]
                    module.weight.data *= mask
                    module.bias.data *= mask

    def mask_gradients(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.pruning_masks:
                    mask = self.pruning_masks[name]
                    if module.weight.grad is not None:
                        module.weight.grad *= mask
                    if module.bias.grad is not None:
                        module.bias.grad *= mask

    def prune_parameters(self, threshold=0.05):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.pruning_masks:
                    abs_w = torch.abs(module.weight)
                    dead_indices = abs_w < threshold
                    if dead_indices.any():
                        self.pruning_masks[name][dead_indices] = 0
