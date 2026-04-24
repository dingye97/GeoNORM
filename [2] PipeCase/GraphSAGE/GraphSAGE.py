import torch
import torch.nn as nn
import torch_geometric.nn as nng


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nb_hidden_layers=2, bn_bool=False):
        super(GraphSAGE, self).__init__()

        self.nb_hidden_layers = nb_hidden_layers
        self.size_hidden_layers = hidden_dim
        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        # 三层 encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 三层 decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # SAGEConv 层
        self.in_layer = nng.SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nng.SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
            for _ in range(nb_hidden_layers - 1)
        ])
        self.out_layer = nng.SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)

        # BatchNorm
        if self.bn_bool:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim, track_running_stats=False)
                for _ in range(nb_hidden_layers)
            ])

    def forward(self, x, mesh, edge_index):
        """
        x: [B, N, C_x]
        mesh: [B, N, C_mesh]
        edge_index: [2, N_edges] (same for all graphs)
        """
        B, N, _ = x.shape

        # 拼接 mesh
        mesh = mesh.repeat(B, 1, 1)
        x = torch.cat((x, mesh), dim=-1)   # [B, N, C_total]

        # flatten batch
        x = x.view(B*N, -1)

        # 复制 edge_index 并偏移
        device = x.device
        edge_index_batch = []
        for b in range(B):
            edge_index_batch.append(edge_index + b*N)
        edge_index_batch = torch.cat(edge_index_batch, dim=1).to(device)  # [2, B*N_edges]

        # 编码
        z = self.encoder(x)

        # SAGEConv
        z = self.in_layer(z, edge_index_batch)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for i, conv in enumerate(self.hidden_layers):
            z = conv(z, edge_index_batch)
            if self.bn_bool:
                z = self.bn[i + 1](z)
            z = self.activation(z)

        z = self.out_layer(z, edge_index_batch)

        # 解码
        z = self.decoder(z)

        # reshape 回 [B, N, out_dim]
        out_dim = z.shape[-1]
        z = z.view(B, N, out_dim)
        return z


    
