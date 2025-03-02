"""
具有对比学习的分层多尺度自适应图形模型
HMAGT-CL模型，我们提出来的新模型

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# 辅助函数：计算 kNN 图
# ------------------------------
def knn_graph(x, k):
    """
    输入：
      x: 节点特征矩阵，尺寸 (N, C)
      k: 邻居数
    输出：
      edge_index: (2, N*k) 张量，其中第一行表示源节点索引，
                  第二行表示目标节点索引
    """
    inner = -2 * torch.matmul(x, x.t())
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distances = xx + inner + xx.t()  
    _, idx = torch.topk(distances, k=k + 1, largest=False)
    idx = idx[:, 1:]  
    N = x.size(0)
    # 构造 edge_index：对每个节点生成 k 边
    src = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, k) 
    edge_index = torch.stack([src, idx], dim=0).view(2, -1) 
    return edge_index


# ------------------------------
# 模块 1：动态图构建
# ------------------------------
class DynamicGraphConstructor(nn.Module):
    """
    动态图构建模块：
      - 将 CNN 提取的特征图 (B, C, H, W) 打平为节点 (B, N, C)，其中 N = H * W；
      - 对每个样本计算 kNN 图，返回边索引列表。
    """

    def __init__(self, k=8):
        super(DynamicGraphConstructor, self).__init__()
        self.k = k
        self.project = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, features):
        B, C, H, W = features.size()
        N = H * W
        proj_feat = self.project(features)  
        nodes = proj_feat.view(B, C, -1).permute(0, 2, 1) 
        edge_index_list = []
        for i in range(B):
            # 对每个样本计算 kNN 图
            x = nodes[i] 
            edge_index = knn_graph(x, self.k)  
            edge_index_list.append(edge_index)
        return nodes, edge_index_list


# ------------------------------
# 模块 2：图卷积和多尺度融合
# ------------------------------
class GraphConvolution(nn.Module):
    """
    图卷积层：对每个节点聚合其邻居特征，然后线性变换。
    """

    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        N = x.size(0)
        agg = torch.zeros_like(x)
        src, tgt = edge_index  
        agg.index_add_(0, src, x[tgt])
        counts = torch.bincount(src, minlength=N).unsqueeze(1).float() + 1e-6
        agg = agg / counts
        out = self.linear(agg)
        return out


class MultiScaleGraphConvolution(nn.Module):
    """
    多尺度图卷积模块：
      - 采用不同尺度（不同 k 值）的邻域信息进行图卷积；
      - 这里我们模拟两种尺度：例如使用前 k1 个邻居和前 k2 个邻居；
      - 利用门控机制融合两种尺度的卷积结果。
    """

    def __init__(self, in_channels, out_channels, scales=[4, 8]):
        super(MultiScaleGraphConvolution, self).__init__()
        self.scales = scales 
        self.convs = nn.ModuleList([GraphConvolution(in_channels, out_channels) for _ in scales])
        self.gate = nn.Sequential(
            nn.Linear(out_channels * len(scales), out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        outputs = []
        N = x.size(0)
        total_edges = edge_index.size(1)
        k_max = total_edges // N
        for scale, conv in zip(self.scales, self.convs):
            if scale > k_max:
                scale = k_max
            # 取前 N*scale 边
            scale_edge = edge_index[:, :N * scale]
            out = conv(x, scale_edge)
            outputs.append(out)
        # 融合：拼接各尺度输出，计算门控权重，然后融合
        multi_scale_feat = torch.cat(outputs, dim=-1)  
        gate_weight = self.gate(multi_scale_feat)  
        # 融合两种尺度：加权求和
        fused = gate_weight * outputs[0] + (1 - gate_weight) * outputs[1]
        return fused


# ------------------------------
# 模块 3：图池化
# ------------------------------
class GraphPooling(nn.Module):
    """
    图池化层：根据节点特征重要性评分选择 top-k 节点
    """

    def __init__(self, in_channels, ratio=0.5):
        super(GraphPooling, self).__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_channels, 1)

    def forward(self, x):
        # x: (N, C)
        scores = self.score_layer(x).squeeze(-1)  
        N = x.size(0)
        k = max(1, int(self.ratio * N))
        _, idx = torch.topk(scores, k, largest=True)
        x_pool = x[idx]
        return x_pool, idx


# ------------------------------
# 模块 4：图变换器层
# ------------------------------
class GraphTransformerLayer(nn.Module):
    """
    图变换器层：利用多头自注意力对图节点进行全局建模。
    输入要求：对于单个图，输入尺寸为 (N, C)，需要调整为 (N, batch=1, C) 形式。
    """

    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_in = x.unsqueeze(1)  
        x_trans = x_in.transpose(0, 1)  #  适应 MultiheadAttention 的输入要求
        attn_out, _ = self.attn(x_trans, x_trans, x_trans)
        x_trans = x_trans + self.dropout(attn_out)
        x_trans = self.norm1(x_trans)
        ff_out = self.ff(x_trans)
        x_trans = x_trans + self.dropout(ff_out)
        x_trans = self.norm2(x_trans)
        # 恢复形状
        x_out = x_trans.transpose(0, 1).squeeze(1)
        return x_out


# ------------------------------
# 模型：Hierarchical Multi-scale Adaptive Graph Transformer with Contrastive Learning (HMAGT-CL)
# ------------------------------
class HMAGT_CL(nn.Module):
    """
    主要创新点：
      1. **动态图构建**：通过 CNN 提取特征后，动态构建图（利用 kNN），获得节点及边信息
      2. **多尺度图卷积**：分别从不同邻域尺度聚合节点信息，并利用门控机制融合
      3. **层次池化**：通过图池化层选择重要节点，形成层次结构，降低图规模
      4. **图变换器**：采用多头自注意力对池化后的图节点进行全局建模

    最终输出全局图表示，经全连接层用于分类
    """

    def __init__(self, num_classes=2, backbone_channels=64, k=8, pool_ratio=0.5):
        super(HMAGT_CL, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, backbone_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 动态图构建模块
        self.graph_constructor = DynamicGraphConstructor(k=k)
        # 多尺度图卷积模块
        self.ms_conv = MultiScaleGraphConvolution(in_channels=backbone_channels, out_channels=backbone_channels,
                                                  scales=[4, 8])
        # 图池化模块
        self.pool = GraphPooling(in_channels=backbone_channels, ratio=pool_ratio)
        # 图变换器层
        self.transformer = GraphTransformerLayer(in_channels=backbone_channels, num_heads=4, dropout=0.1)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(backbone_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        features = self.backbone(x)  
        outputs = []
        nodes_all, edge_index_list = self.graph_constructor(features)
        for i in range(B):
            nodes = nodes_all[i]
            edge_index = edge_index_list[i]  
            # 多尺度图卷积
            nodes_ms = self.ms_conv(nodes, edge_index)  
            # 图池化
            nodes_pool, _ = self.pool(nodes_ms) 
            # 图变换器，全局建模
            nodes_trans = self.transformer(nodes_pool) 
            # 全局平均池化，得到图表示 (C,)
            graph_repr = nodes_trans.mean(dim=0)
            outputs.append(graph_repr)
        outputs = torch.stack(outputs, dim=0)  
        logits = self.classifier(outputs)  
        return logits


if __name__ == "__main__":
    # 单元测试
    model = HMAGT_CL(num_classes=2, backbone_channels=64, k=8, pool_ratio=0.5)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print("输出形状:", logits.shape)  
