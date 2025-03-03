本项目演示了在医疗影像（皮肤癌数据）场景下，如何使用 **联邦学习** 与 **同态加密**（基于 TenSEAL）的综合解决方案，包含以下关键功能：

1. **数据加载与预处理** (`dataloader.py`)
2. **同态加密与分块加密** (`he_encryption.py`)
3. **深度学习模型（包含创新的图结构与激活函数等）** (`models.py`)
4. **联邦学习主流程** (`federated_learning.py`)

## 项目结构

```
.
├── dataloader.py
├── federated_learning.py
├── he_encryption.py
├── models.py
└── README.md
```

- **dataloader.py**  
  包含 `SkinCancerDataset` 类与相关加载逻辑，将皮肤癌数据 CSV（元数据）与实际图像文件对接并输出 `torch.utils.data.Dataset`。
- **federated_learning.py**  
  - 项目入口，定义联邦学习主流程。  
  - 初始化全局模型、同态加密上下文、服务器和多个客户端。  
  - 进行若干轮联邦学习，每个客户端训练后加密上传更新，服务器同态聚合并解密更新全局模型。
- **he_encryption.py**  
  - `DynamicHE`：核心的同态加密管理类，基于 TenSEAL CKKS。  
    - 支持 **分块加密**（`encrypt_chunked` / `decrypt_chunked`）处理大数据；  
    - 支持 **动态调参** 和 **激活函数近似**（`encrypted_relu`）；  
    - 提供 **同态聚合** 与 **模拟重缩放** 等功能。  
  - `HEEncryptor`：面向联邦学习的加密处理器，用于对模型参数进行同态加密与安全聚合。
- **models.py**  
  - 定义了多个深度学习模型，包括 **AMGT-CL** 或 **HMAGT** 等创新结构，用于图结构、Transformer等。  

## 准备工作

**数据集下载**

通过网盘分享的文件：ISIC 2020 JPG 256x256 RESIZED.zip
链接: https://pan.baidu.com/s/1M6iOPSqhaNx4AOgE-dZCuQ?pwd=wwtr 提取码: wwtr 


**安装依赖**  

- Python >= 3.7  
- PyTorch 与 torchvision  
- [TenSEAL](https://github.com/OpenMined/TenSEAL)  
- tqdm

```bash
pip install torch torchvision tqdm
pip install tenseal
```

## 如何运行

1. **放置数据集**  

   - 数据集路径为 `dataset/train-metadata.csv` 和 `dataset/train-image`。  
   - 在 `federated_learning.py` 中修改相应的 `csvfile` 与 `imgdir` 路径。

2. **启动联邦学习**  
   在项目根目录执行：

   ```bash
   python federated_learning.py
   ```

