# Vit-Backbone

Welcome to \`**Vit-Backbone**\` !

> @Author: Jintu Zheng
>
> @Email: jintuzheng@outlook.com

Now release version is 1.0.01 

> Fixed Update in 05/07/2022



## :smile:Introduction

It provides:

- Newly and powerful Vision Transformer's Backbones (all with pretrained weights) simple usage
- You can build your own model quickly based on Vision Transformer backbone using **vitbackbone**
- More ViT backbones are coming soon......:cupid:



This repository contains an PyTorch reimplementation of serval bacbones:

> Format means (\`Pretrained Dataset Name\`) # img_size = \`resolution\`

- **Swin-Transformer**
- - **swin_transformer_tiny** (ImageNet) # img_size = 224
  - **swin_transformer_tiny** (ImageNet+ADE20K) # img_size = 512
  - **swin_transformer_small** (ImageNet) # img_size=224
  - **swin_transformer_small** (ImageNet+ADE20K) # img_size = 512
  - **swin_transformer_basic** (ImageNet)
  - **swin_transformer_basic** (ImageNet)
  - **swin_transformer_large**
  - **swin_transformer_large**
- **VAN**
- - van_tiny
  - van_small
  - van_base
  - van_large
- **FocalNet**
- - ......coming soon



## Quick Start

### Step1: Install

```bash
pip install vitbackbone
```



### Step2: Pretrained Download

- **Baidu Pan ..**

  Download and put all weights(you can also download the special one you need) in a folder named `weights`. Then if you are initing a model, first you should do as followed:

  ```python
  from vitbackbone import vitmodels as vitb
  
  # Using a local weights package
  models = vitb('/home/xxx/weights') 
  ```

  

  The folder's format must be as followed:

  ```
  weights 
  ├── swin
  │   ├── swin_base_patch4_window12_384.pth
  │   ├── swin_large_patch4_window12_384_22kto1k.pth
  │   └── ...
  └── van
      ├── van_ham_base.pth
      ├── van_ham_large.pth
      └── ...
  ```

- **Tricks**:

  If you are our yunlab member, the special usage in our server:

  ```python
  from vitbackbone import vitmodels as vitb
  
  # Using a local weights package
  models = vitb('ylab') 
  ```

  

### **Demo Usage 1**

  ```python
import torch
from vitbackbone import vitmodels as vitb

# Using a local weights package
models = vitb('/home/xxx/weights') 

# [1] Using ImageNet pretrained weight
model = models.swin_transformer_tiny(pretrained = True, pretrain_type = 0)

# [2] Using ImageNet+ADE20k pretrained weight
# model = vit_models.swin_transformer_tiny(pretrained = True, pretrain_type = 1)

# [3] No using pretrained weight
# model = vit_models.swin_transformer_tiny(pretrained = False)

model.cuda()
model.eval()

with torch.no_grad():
    x = torch.randn([1,3,224,224]).cuda()
    outs = model(x) # return different stages output as a tensors list


  ```

### Demo Usage2

  ```python
import torch
from vitbackbone import vitmodels as vitb

# Using a local weights package
models = vitb('ylab') 

# [1] Using ImageNet pretrained weight
model = models.swin_transformer_tiny(pretrained = True, pretrain_type = 0)

model.cuda()
model.eval()

with torch.no_grad():
    x = torch.randn([1,3,224,224]).cuda()
    outs = model(x) # return different stages output as a tensors list


  ```

  

## Thanks

This codebase is built based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Focal Transformer](https://github.com/microsoft/Focal-Transformer), [FocalNet](https://github.com/microsoft/FocalNet), and [VAN-Classification](https://github.com/Visual-Attention-Network/VAN-Classification)

Thank the authors for the nicely organized code!

  
