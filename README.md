# SGP: Towards Efficient Test-time Adaptation via Sensitivity-Guided Pruning

This repository is the official implementation of SGP: Towards Efficient Test-time Adaptation via Sensitivity-Guided Pruning. 

This work is currently submitted to TPAMI and is an extended version of our CVPR 2025 paper: *Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning*. We are gradually open-sourcing the code here.

## Abstract
Continual test-time adaptation (CTTA) aims to online adapt a source pre-trained model to non-stationary data streams during inference. 
Most existing CTTA methods prioritize adaptation effectiveness while largely overlooking computational efficiency, which is critical as the inherently online process of CTTA often clashes with the constraints of resource-limited scenarios. 
In this paper, we propose SGP, an efficient CTTA framework via Sensitivity-Guided Pruning. 
Our motivation stems from the empirical observation that not all learned source features are beneficial for the target domain; certain domain-sensitive features can negatively impact target performance. 
Inspired by this, we introduce a sensitivity-guided pruning strategy that quantifies each feature based on its sensitivity to domain discrepancies and applies weighted sparsity regularization to selectively suppress these sensitive features. 
By focusing adaptation efforts on domain-invariant features, SGP simultaneously reduces computational overhead and stabilizes the adaptation process. 
Furthermore, a stochastic reactivation mechanism is incorporated to randomly restore pruned features, allowing the model to reassess their utility in changing environments. 
Extensive experiments across nine benchmarks on three distinct tasks (classification, detection, and segmentation) demonstrate that SGP achieves superior performance across both CNN and ViT architectures, reducing computational FLOPs by up to 13.0% compared to SOTA methods.

---

## Prerequisite

### 1. Data and Checkpoints Download
Please download the datasets and pre-trained checkpoints from the links below:

* **CIFAR10-C / CIFAR100-C / ImageNet-C Datasets:** [Baidu Netdisk](https://pan.baidu.com/s/19MJRlaSWDQMYpBF1ppPzhA) (Pwd: `44t2`)
* **Pre-trained Checkpoints:** [Baidu Netdisk](https://pan.baidu.com/s/188I6LYL8wWVkQPDZ7VqEDQ) (Pwd: `tpxr`)

### 2. Directory Structure
You need to download the dataset and weight archives and extract them into the `resource` folder. The correct folder structure should be as follows:

```text
.
├── pruningTTA_CNN
├── pruningTTA_ViT
├── resource
│   ├── CIFAR-10-C
│   ├── CIFAR-100-C
│   ├── ImageNet-C
│   ├── vit_base_patch16_224.augreg_in21k_ft_in1k
│   ├── vit_base_patch16_384.augreg_in21k_ft_in1k
│   ├── Hendrycks2020AugMix_ResNeXt.pt
│   ├── pretrain_cifar100.t7
│   ├── Standard.pt
│   └── vit_base_384_cifar10.t7
...

## Usage and Experiments

### 1. CNN-based Experiments

First, navigate to the CNN directory and install dependencies:

```bash
cd pruningTTA_CNN
pip install -i [https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/) -r requirements.txt
```

Run adaptation on different benchmarks:

**CIFAR10-to-CIFAR10C:**
```bash
python test_time.py --cfg cfgs/cifar10_c/pruning.yaml
```

**CIFAR100-to-CIFAR100C:**
```bash
python test_time.py --cfg cfgs/cifar100_c/pruning.yaml
```

**ImageNet-to-ImageNetC:**
```bash
python test_time.py --cfg cfgs/imagenet_c/pruning.yaml
```

---

### 2. ViT-based Experiments

#### CIFAR Benchmarks

First, navigate to the ViT CIFAR directory and install dependencies:

```bash
cd pruningTTA_ViT/cifar
pip install -i [https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/) -r requirements.txt
```

Run adaptation on different benchmarks:

**CIFAR10-to-CIFAR10C:**
```bash
python cifar10c_vit.py --cfg cfgs/cifar10/pruning.yaml
```

**CIFAR100-to-CIFAR100C:**
```bash
python cifar100c_vit.py --cfg cfgs/cifar100/pruning.yaml
```

#### ImageNet Benchmark

First, navigate to the ViT ImageNet directory and install dependencies:

```bash
cd pruningTTA_ViT/imagenet
pip install -i [https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/) -r requirements.txt
```

Run adaptation on the benchmark:

**ImageNet-to-ImageNetC:**
```bash
python imagenetc.py --cfg cfgs/pruning.yaml
```
