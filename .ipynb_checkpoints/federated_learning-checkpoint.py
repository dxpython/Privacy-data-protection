import logging
import time
import traceback

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import tenseal as ts

# 项目内部模块
from dataloader import SkinCancerDataset
from he_encryption import DynamicHE, HEEncryptor
from models import HMAGT_CL
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# -------------------------------
# 配置日志
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FedLearn")

# -------------------------------
# 全局配置
# -------------------------------
config = {
    # 客户端数量、学习率
    "num_clients": 3,
    "client_lr": 0.001,
    "client_batch_size": 32,
    "client_num_workers": 4,

    # 模型参数
    "num_classes": 2,
    "backbone_channels": 64,
    "k": 8,
    "pool_ratio": 0.5,

    # 同态加密参数
    "init_poly_degree": 32768,
    "init_scale": 2**20,

    # 联邦学习轮次
    "rounds": 3,

    # 数据集
    "csv_file": "/root/autodl-tmp/HE/dataset/train-metadata.csv",
    "img_dir": "/root/autodl-tmp/HE/dataset/train-image/image"
}

# -------------------------------
# 评价指标计算函数
# -------------------------------
def evaluate_metrics(model, dataloader, device):
    """
    计算 Accuracy, Precision, Recall, F1 四个指标。
    """
    model.eval()
    correct = 0
    total = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                if p == 1 and l == 1:
                    TP += 1
                elif p == 1 and l == 0:
                    FP += 1
                elif p == 0 and l == 1:
                    FN += 1
                else:
                    TN += 1

    # 计算四大指标
    acc = correct / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return acc, precision, recall, f1

# -------------------------------
# 客户端类
# -------------------------------
class Client:
    def __init__(self, client_id, dataloader, device, lr):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr

        # 使用 HMAGT-CL 模型
        self.model = HMAGT_CL(num_classes=config["num_classes"],
                              backbone_channels=config["backbone_channels"],
                              k=config["k"],
                              pool_ratio=config["pool_ratio"]).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self):
        logger.info(f"[Client {self.client_id}] 开始本地训练 (LR={self.lr})")
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in tqdm(self.dataloader,
                                   desc=f"Client {self.client_id} Training",
                                   leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"[Client {self.client_id}] 本地训练完成，平均损失: {avg_loss:.4f}")

    def evaluate_local(self):
        """
        用训练集(或本地验证集)计算四大指标
        """
        acc, prec, rec, f1 = evaluate_metrics(self.model, self.dataloader, self.device)
        logger.info(f"[Client {self.client_id}] 本地评估 -> "
                    f"ACC={acc:.4f} | PREC={prec:.4f} | REC={rec:.4f} | F1={f1:.4f}")

    def get_model_update(self, global_model):
        """
        计算模型更新：返回本地模型参数与全局模型参数之差
        """
        local_vec = parameters_to_vector(self.model.parameters())
        global_vec = parameters_to_vector(global_model.parameters())
        update = local_vec - global_vec
        return update.detach().cpu().numpy()

    def set_model(self, global_model):
        """
        将全局模型参数复制到本地模型中
        """
        global_vec = parameters_to_vector(global_model.parameters())
        vector_to_parameters(global_vec, self.model.parameters())

# -------------------------------
# 服务器类
# -------------------------------
class Server:
    def __init__(self, global_model, device, he):
        self.global_model = global_model
        self.device = device
        self.he = he  # DynamicHE 实例
        self.encryptor = HEEncryptor(self.he.context)

    def aggregate_updates(self, encrypted_updates):
        logger.info("[Server] 同态聚合客户端更新")
        aggregated = encrypted_updates[0].copy()
        for update in encrypted_updates[1:]:
            aggregated += update
        aggregated *= (1.0 / len(encrypted_updates))
        return aggregated

    def update_global_model(self, update_vector):
        global_vec = parameters_to_vector(self.global_model.parameters())
        update_tensor = torch.tensor(update_vector,
                                     dtype=global_vec.dtype,
                                     device=self.device)
        new_global_vec = global_vec + update_tensor
        vector_to_parameters(new_global_vec, self.global_model.parameters())
        logger.info("[Server] 全局模型参数已更新。")


# -------------------------------
# 联邦学习主流程
# -------------------------------
def main():
    logger.info("=" * 30 + " FEDERATED LEARNING START " + "=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"当前设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    csvfile = config["csv_file"]
    imgdir = config["img_dir"]
    try:
        dataset = SkinCancerDataset(csv_file=csvfile,
                                    img_dir=imgdir,
                                    transform=transform)
        logger.info(f"数据集加载成功: {len(dataset)} 条样本")
    except Exception as e:
        logger.error("加载数据集时出现异常：")
        logger.error(traceback.format_exc())
        return

    # 划分客户端数据
    num_clients = config["num_clients"]
    lengths = [len(dataset) // num_clients for _ in range(num_clients)]
    lengths[-1] += len(dataset) - sum(lengths)
    client_datasets = random_split(dataset, lengths)
    client_loaders = [DataLoader(ds,
                                 batch_size=config["client_batch_size"],
                                 shuffle=True,
                                 num_workers=config["client_num_workers"])
                      for ds in client_datasets]

    # 初始化全局模型
    global_model = HMAGT_CL(num_classes=config["num_classes"],
                            backbone_channels=config["backbone_channels"],
                            k=config["k"],
                            pool_ratio=config["pool_ratio"]).to(device)

    # 初始化同态加密系统
    try:
        he = DynamicHE(init_poly_degree=config["init_poly_degree"],
                       init_scale=config["init_scale"])
    except Exception as ex:
        logger.error("同态加密初始化失败，程序终止：")
        logger.error(traceback.format_exc())
        return

    # 创建客户端与服务器
    clients = [Client(client_id=i,
                      dataloader=client_loaders[i],
                      device=device,
                      lr=config["client_lr"])
               for i in range(num_clients)]
    server = Server(global_model=global_model, device=device, he=he)

    # 联邦学习
    rounds = config["rounds"]
    logger.info(f"开始联邦学习，共 {rounds} 轮...")
    total_start = time.time()

    for r in range(rounds):
        round_start = time.time()
        logger.info(f"=== 轮次 {r+1} 开始 ===")

        encrypted_updates = []
        for client in clients:
            logger.info(f"[Client {client.client_id}] 设置全局模型")
            client.set_model(global_model)

            logger.info(f"[Client {client.client_id}] 训练开始")
            client_train_start = time.time()
            client.train_one_epoch()
            client.evaluate_local()  
            client_train_end = time.time()

            train_spent = client_train_end - client_train_start
            logger.info(f"[Client {client.client_id}] 训练耗时 {train_spent:.2f} 秒")

            update = client.get_model_update(global_model)

            # 加密更新
            try:
                enc_update = ts.ckks_vector(he.context, update.tolist())
                encrypted_updates.append(enc_update)
            except Exception as ee:
                logger.error(f"[Client {client.client_id}] 加密更新失败：")
                logger.error(traceback.format_exc())
                return

        # 服务器同态聚合
        logger.info("[Server] 同态聚合客户端更新")
        agg_enc = server.aggregate_updates(encrypted_updates)

        # 解密并更新全局模型
        try:
            agg_update = he.decrypt(agg_enc)
            server.update_global_model(agg_update)
        except Exception as e:
            logger.error("[Server] 解密或更新全局模型失败：")
            logger.error(traceback.format_exc())
            return

        round_end = time.time()
        logger.info(f"=== 轮次 {r+1} 结束，耗时 {round_end - round_start:.2f} 秒 ===")

    total_end = time.time()
    logger.info(f"联邦学习训练全部完成，总耗时 {total_end - total_start:.2f} 秒。")
    logger.info("=" * 30 + " FEDERATED LEARNING END " + "=" * 30)


if __name__ == "__main__":
    main()
