import torch
import torch.nn as nn
from base_model import BaseModel
from neighbor_ops import NeighborSearchLayer, NeighborMLPConvLayer

from fno import FNO2d

from net_utils import PositionalEmbedding, MLP

import os
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_LOGS"] = "default:ERROR"
class GNOFNOGNO(BaseModel):
    def __init__(
            self,
            radius_in=0.05,
            radius_out=0.05,
            embed_dim=64,
            hidden_channels=(86, 86),
            in_channels=1,
            out_channels=1,
            fno_out_channels=86,
            modes = 32,
            width = 64
    ):
        super().__init__()
     
        self.nb_search_in = NeighborSearchLayer(radius_in)
        self.nb_search_out = NeighborSearchLayer(radius_out)
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.fea_embed = MLP(
            [in_channels, embed_dim, 2 * embed_dim], torch.nn.GELU
        )
       

        kernel1 = MLP(
            [6 * embed_dim, 256, 128, hidden_channels[0]], torch.nn.GELU
        )
        self.gno1 = NeighborMLPConvLayer(mlp=kernel1)

     
        kernel2 = MLP(
            [fno_out_channels + 4 * embed_dim, 256, 128, hidden_channels[1]], torch.nn.GELU
        )
        self.gno2 = NeighborMLPConvLayer(mlp=kernel2)


        self.fno = FNO2d(2*embed_dim+hidden_channels[0],fno_out_channels,modes, modes, width)



        
        self.projection = nn.Sequential(
                nn.Linear(hidden_channels[0], hidden_channels[0]),
                nn.GELU(),
                nn.Linear(hidden_channels[0], out_channels)
            )

    # x_in : (n_in, 2)
    # x_latent : (n_x, n_y, n_z, 3)    
    # u_fea: (n_in, 3)
    def forward(self, x_in, x_latent, u_fea):
        #u_fea = u_fea.squeeze(0) 
        batch_zise = u_fea.shape[0]
        # manifold to latent neighborhood

        in_to_latent_nb = self.nb_search_in(x_in, x_latent.view(-1, 2))
        

        # latent to manifold neighborhood
        latent_to_in_nb = self.nb_search_out(x_latent.view(-1, 2), x_in)

   
        resolution = x_latent.shape[0]  
        
        n_in = x_in.shape[0]
        
        
        x_in_embed = self.pos_embed(
            x_in.reshape(-1, )
        ).reshape(
            (n_in, -1)
        ) 
        x_in_embed = x_in_embed.repeat(batch_zise, 1, 1)
        
        # Embed latent space features
        fea_embed = self.fea_embed(u_fea).reshape(
            (batch_zise, n_in, -1)
        )  # (n_in, 2*embed_dim) ([1312, 128])

        combinedx_in = torch.cat([x_in_embed, fea_embed], dim=-1)  

        # Embed latent space coordinates
        x_latent_embed = self.pos_embed(
            x_latent.reshape(-1, )
        ).reshape(
            (resolution ** 2, -1)
        )  
        x_latent_embed = x_latent_embed.repeat(batch_zise, 1, 1)
        
        # GNO : project to latent space
        u = self.gno1(
            combinedx_in, in_to_latent_nb, x_latent_embed)  
       
 
        u = (
            u.reshape(batch_zise,resolution, resolution, -1).permute(0,3, 1, 2)
        )  
    
        # Add positional embedding
        x_latent_embed = (
            x_latent_embed.reshape(batch_zise, resolution, resolution, -1).permute(0,3, 1, 2)
        )  
 
        u = torch.cat(
            (x_latent_embed, u), dim=1
        ) 

        # FNO on latent space
        u = self.fno(u)  # (1, n_x, n_y, fno_out_channels)
            

        u = (
            u.reshape(batch_zise,resolution ** 2, -1)
        ) 

        # GNO : project to manifold
        u = self.gno2(u, latent_to_in_nb, combinedx_in)  


        # Pointwise projection to out channels
        u = self.projection(u) 

        return u





