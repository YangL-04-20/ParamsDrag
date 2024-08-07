import sys
import torch
import torch.nn as nn

from training.networks import *

#----------------------------------------------------------------------------
class Paramap(torch.nn.Module):
    def __init__(self,
        dsp,
        dvp,
        dspe = 512,
        dvpe = 512,
    ):
        super().__init__()
        # simulation parameters subnet
        self.sparams_subnet = nn.Sequential(
            nn.Linear(dsp, dspe), nn.LeakyReLU(),
            nn.Linear(dspe, dspe), nn.LeakyReLU()
        )
        # view parameters subnet
        self.vparams_subnet = nn.Sequential(
            nn.Linear(dvp, dvpe), nn.LeakyReLU(),
            nn.Linear(dvpe, dvpe), nn.LeakyReLU()
        )
        # merged parameters subnet
        self.mparams_subnet = nn.Sequential(
            nn.Linear(dspe + dvpe, 512, bias=False), nn.LeakyReLU()
        )
    
    def forward(self, sp, vp):
        sp = self.sparams_subnet(sp)
        vp = self.vparams_subnet(vp)
        mp = torch.cat((sp, vp), 1)
        mp = self.mparams_subnet(mp)
        
        return mp

#----------------------------------------------------------------------------

class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        dsp,
        dvp,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        c_dim               = 0,    # Conditioning label (C) dimensionality.
        img_channels        = 3,    # Number of output color channels.
        **synthesis_kwargs2,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        if len(synthesis_kwargs) == 0:
            synthesis_kwargs = synthesis_kwargs2
        self.paramap = Paramap(dsp, dvp)
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, sp, vp, truncation_psi=1, truncation_cutoff=None, update_emas=False,
                return_parasub=False,
                return_ws=False,
                return_feature=False, **synthesis_kwargs):
        
        z = self.paramap(sp, vp)
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, return_feature=return_feature, **synthesis_kwargs)

        if return_parasub:
            return img, z
        if return_ws:
            return img, ws
        else:
            return img