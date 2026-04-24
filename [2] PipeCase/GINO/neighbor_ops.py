import torch
from torch.nn import functional as F
from torch_scatter import segment_csr
import os
# os.environ["PYTORCH_JIT_LOG_LEVEL"] = "ERROR"
# os.environ["TORCH_LOGS"] = "default:ERROR"
class NeighborSearchLayer(torch.nn.Module):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius

    def forward(self, inp_positions: torch.Tensor, out_positions: torch.Tensor):
        """
        使用torch.cdist实现固定半径邻居搜索
        
        参数:
            inp_positions: [N, 3] 输入点坐标
            out_positions: [M, 3] 查询点坐标
            
        返回:
            包含邻居索引和row_splits的字典
        """
        # 计算所有点对之间的距离
        distances = torch.cdist(out_positions, inp_positions)  # [M, N]
        
        # 找出距离小于radius的邻居
        mask = distances <= self.radius
        edge_index = torch.nonzero(mask, as_tuple=False).T  # [2, num_edges]
        # print('inp_positions',inp_positions.shape)
        # print('edge_index',edge_index.shape)
        
        # 转换为类似FixedRadiusSearchResult的结构
        neighbors_index = edge_index[1]  # 邻居索引
        query_index = edge_index[0]      # 查询点索引
        
        # 计算row_splits (类似CSR格式的指针)
        M = out_positions.shape[0]
        counts = torch.bincount(query_index, minlength=M)
        row_splits = F.pad(torch.cumsum(counts, dim=0), (1, 0))  # [M+1]
        
        return {
            'neighbors_index': neighbors_index,
            'neighbors_row_splits': row_splits
        }



import torch
from torch import nn


class NeighborMLPConvLayer(nn.Module):
    def __init__(self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(2 * in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_channels)
            )
        self.mlp = mlp

    def forward(self, in_features, neighbors, out_features=None, use_vmap=True):
        """
        参数:
            in_features: [B, N, C]，B个定义在相同图结构上的输入特征
            neighbors: 包含 CSR 结构的 dict，字段：
                - neighbors_index: [E]
                - neighbors_row_splits: [N+1]
            out_features: [B, N, C]（可选，默认为 in_features）
            use_vmap: 是否使用 vmap 加速（要求 PyTorch ≥ 2.0）
        返回:
            [B, N, C]，batch 输出特征
        """
        if out_features is None:
            out_features = in_features

        if use_vmap and hasattr(torch, 'vmap'):
            return torch.vmap(self._single_forward, in_dims=(0, 0, None))(in_features, out_features, neighbors)
        else:
            outputs = []
            B = in_features.shape[0]
            for i in range(B):
                out_i = self._single_forward(in_features[i], out_features[i], neighbors)
                outputs.append(out_i)
            return torch.stack(outputs, dim=0)

    def _single_forward(self, in_feat, out_feat, neighbors):
        """
        单个样本的处理逻辑
        参数:
            in_feat: [N, C]
            out_feat: [N, C]
            neighbors: dict，包含CSR结构
        返回:
            [N, C]
        """
        rep_features = in_feat[neighbors['neighbors_index'].long()]  # [E, C]
        rs = neighbors['neighbors_row_splits'].long()  # [N+1]
        num_reps = rs[1:] - rs[:-1]  # [N]
        self_features = torch.repeat_interleave(out_feat, num_reps, dim=0)  # [E, C]

        agg_features = torch.cat([rep_features, self_features], dim=1)  # [E, 2C]
        rep_features = self.mlp(agg_features)  # [E, C_out]

        out_features = self._segment_csr(rep_features, rs, reduce=self.reduction)  # [N, C_out]
        return out_features

    def _segment_csr(self, src, indptr, reduce='mean'):
        """
        使用 torch_scatter.segment_csr 实现的高效CSR分段聚合函数
        参数:
            src: [E, C]
            indptr: [N+1]
            reduce: 'sum', 'mean', 等
        返回:
            [N, C]
        """
        return segment_csr(src, indptr, reduce=reduce)




# class NeighborMLPConvLayer(torch.nn.Module):
#     def __init__(self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"):
#         super().__init__()
#         self.reduction = reduction
#         if mlp is None:
#             mlp = torch.nn.Sequential(
#                 torch.nn.Linear(2 * in_channels, hidden_dim),
#                 torch.nn.GELU(),
#                 torch.nn.Linear(hidden_dim, out_channels)
#             )
#         self.mlp = mlp



    # def forward(self, in_features, neighbors, out_features=None):
    #     """
    #     参数:
    #         in_features: [N, C] 输入节点特征
    #         neighbors: 包含neighbors_index和neighbors_row_splits的字典
    #         out_features: [M, C] 输出节点特征(可选)
    #     """
    #     if out_features is None:
    #         out_features = in_features

    #     # 获取邻居特征
    #     rep_features = in_features[neighbors['neighbors_index'].long()]
        
    #     # 计算每个中心节点的邻居数量
    #     rs = neighbors['neighbors_row_splits']
    #     num_reps = rs[1:] - rs[:-1]
        
    #     # 重复中心节点特征以匹配邻居数量
    #     self_features = torch.repeat_interleave(out_features, num_reps, dim=0)
        
    #     # 拼接特征并通过MLP
    #     agg_features = torch.cat([rep_features, self_features], dim=1)
        
    #     rep_features = self.mlp(agg_features)
        
    #     # 分段聚合
    #     out_features = self._segment_csr(rep_features, 
    #                                    neighbors['neighbors_row_splits'], 
    #                                    reduce=self.reduction)
    #     return out_features


    # def _segment_csr(self, src, indptr, reduce='mean'):
    #     # """
    #     # 自定义CSR分段聚合函数
    #     # src: [num_edges, C] 要聚合的数据
    #     # indptr: [M+1] 分段指针
    #     # reduce: 'sum'或'mean'
    #     # """
    #     # M = indptr.shape[0] - 1
    #     # out = torch.zeros((M, src.shape[1]), device=src.device)
        
    #     # for i in range(M):
    #     #     start, end = indptr[i], indptr[i+1]
    #     #     if reduce == 'sum':
    #     #         out[i] = src[start:end].sum(dim=0)
    #     #     elif reduce == 'mean':
    #     #         out[i] = src[start:end].mean(dim=0) if end > start else 0
                
    #     # return out
    #     """
    #     使用 torch_scatter.segment_csr 实现的高效CSR分段聚合函数
        
    #     参数:
    #         src: [num_edges, C] 要聚合的数据
    #         indptr: [M+1] 分段指针
    #         reduce: 'sum', 'mean', 'min', 'max' 等
        
    #     返回:
    #         out: [M, C] 聚合后的特征
    #     """
    #     return segment_csr(src, indptr, reduce=reduce)