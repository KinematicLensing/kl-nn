import torch
from torch import nn
import math
import normflows as nf
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.nn.nets import ResidualNet

import config
from utils import resolve_feature_index

### Main Network ###
class ForkCNN(nn.Module):
    '''
    Main network consisting of feature extraction branches for images and spectra,
    followed by either point estimate or density estimate layers.
    '''
    def __init__(self, 
                 mode=0,    # 0 = point estimate, 1 = density estimate, 2 = density estimate with TF prior
                 batch_size=config.train['batch_size'],
                 nfeatures=config.train['feature_number'],
                 nspec=config.data['nspec'],
                 # Lognormal TF prior parameters (only used when mode == 2)
                 vcirc_dex=0.05,   # scatter in dex; fixed, represents TF relation scatter
                 vcirc_min=config.par_ranges.get('vcirc', [60.0, 540.0])[0],
                 vcirc_max=config.par_ranges.get('vcirc', [60.0, 540.0])[1],
                 vcirc_idx=None):

        self.bs = batch_size
        self.nfeatures = nfeatures
        self.nspecs = nspec
        if mode in (0, 1, 2):
            self.mode = mode
        else:
            raise ValueError('Mode must be 0 (point estimate), 1 (density estimate), or 2 (density estimate with TF prior)!')

        # Lognormal TF prior settings (only used when mode == 2)
        # dex is fixed (TF scatter); mu is supplied per-galaxy at runtime from magnitude measurements
        self.vcirc_log_scale = vcirc_dex * torch.log(torch.tensor(10.)).item()  # convert dex -> natural-log std
        self.vcirc_min = float(vcirc_min)
        self.vcirc_max = float(vcirc_max)
        self.vcirc_jac = 0.5 * (self.vcirc_max - self.vcirc_min)  # |dv/dx| for x in [-1, 1]
        if vcirc_idx is None:
            feature_names = config.train.get('feature_names', None)
            if feature_names is None:
                raise ValueError("config.train['feature_names'] is required to infer vcirc_idx")
            vcirc_idx = resolve_feature_index(feature_names, 'vcirc', aliases=('v_circ',))
        self.vcirc_idx = int(vcirc_idx)
        if self.vcirc_idx < 0 or self.vcirc_idx >= self.nfeatures:
            raise ValueError(
                f'vcirc_idx={self.vcirc_idx} is out of bounds for nfeatures={self.nfeatures}'
            )

        super(ForkCNN, self).__init__()
        

        # Vision Transformer for image feature extraction
        # self.img_net = VisionTransformer(in_channels=1, 
        #                              embed_dim=512, 
        #                              img_size=48, 
        #                              patch_size=6, 
        #                              num_layers=6, 
        #                              num_heads=8, 
        #                              mlp_ratio=4.0, 
        #                              dropout=0.1)
        self.img_net = ImgCNN()
                                     
        # CNN + RNN for spectra feature extraction
        # self.spec_net = LargeSpecRNN(self.nspecs)
        self.spec_net = SpecCNN(self.nspecs)


        # Define point estimate or density estimate layers
        if self.mode == 0:
            ### Fully-connected layers
            self.fully_connected_layer = nn.Sequential(
                # make sure the first number is equal to the sum of final # of channels in both img and spec branches
                nn.Linear(1024, self.nfeatures),
            )
            self.loss = nn.MSELoss()
        elif self.mode >= 1:
            # Normalizing flow for density estimation
            self.layer_norm = nn.LayerNorm(1024)
            self.setup_flows()
            self.flow = Flow(self.transform, self.base)
            # with torch.no_grad():
            #     for param in self.flow.parameters():
            #         param.zero_()  # Initialize flow to identity

    
    def forward(self, x, y, true, fp=None):
        '''
        x: image tensor
        y: spectrum tensor
        true: target tensor of shape (batch, nfeatures)
        fp: fiber position tensor of shape (batch, nspecs, 2)
        '''
        # Feature extraction from img and spec
        x = nn.functional.normalize(x, dim=[2,3])
        y = nn.functional.normalize(y, dim=[2,3])
        x = self.img_net(x)
        y = self.spec_net(y)

        # Flatten and concatenate
        x = x.view(int(self.bs),-1)
        y = y.view(int(self.bs),-1)
        z = torch.cat((x, y), -1)
        if fp is not None:
            fp = fp.view(fp.size(0), -1)  # Flatten fiber positions
            z = torch.cat((z, fp), dim=-1)

        # Point/density estimate
        if self.mode == 0:
            z = self.fully_connected_layer(z)
            loss = self.loss(z, true)
        else:
            z = self.layer_norm(z)
            loss = -self.flow.log_prob(true, context=z).mean()

        return loss
    
    def extract_latent(self, x, y, true, fp=None):
        '''
        Run through feature extraction but map from true parameters to latent space in flow
        '''
        # Feature extraction from img and spec
        x = nn.functional.normalize(x, dim=[2,3])
        y = nn.functional.normalize(y, dim=[2,3])
        x = self.img_net(x)
        y = self.spec_net(y)

        # Flatten and concatenate
        x = x.view(int(self.bs),-1)
        y = y.view(int(self.bs),-1)
        z = torch.cat((x, y), -1)
        if fp is not None:
            fp = fp.view(fp.size(0), -1)  # Flatten fiber positions
            z = torch.cat((z, fp), dim=-1)

        z = self.layer_norm(z)
        latent = self.flow.transform_to_noise(true, context=z)

        return latent

    def _norm_to_vcirc(self, v_norm):
        v_norm = v_norm.clamp(min=-1.0, max=1.0)
        v_circ = self.vcirc_min + 0.5 * (v_norm + 1.0) * (self.vcirc_max - self.vcirc_min)
        return v_circ.clamp(min=1e-8)

    def _tf_log_prob_from_vnorm(self, v_norm, vcirc_mu):
        """
        v_norm: normalized vcirc in [-1, 1], shape (...)
        vcirc_mu: TF prior center in km/s, broadcastable to v_norm
        """
        v_circ = self._norm_to_vcirc(v_norm)
        mu = vcirc_mu.to(device=v_circ.device, dtype=v_circ.dtype).clamp(min=1e-8)
        prior = torch.distributions.LogNormal(
            loc=torch.log(mu),
            scale=torch.full_like(v_circ, self.vcirc_log_scale)
        )
        return prior.log_prob(v_circ)
        # return prior.log_prob(v_circ) + torch.log(torch.full_like(v_circ, self.vcirc_jac))

    def _flow_v_marginal_from_grid(self, flow_log_prob, v_norm_grid):
        """
        Approximate log P_flow(v_circ | data) by summing over candidates that share v_circ.
        Expects zz to be a structured grid where each v value appears across many (g1, g2).
        flow_log_prob: (batch_size, N)
        v_norm_grid: (N,)
        """
        _, inverse = torch.unique(v_norm_grid, sorted=True, return_inverse=True)
        num_unique = int(inverse.max().item()) + 1
        batch_size = flow_log_prob.shape[0]

        log_p_v_unique = flow_log_prob.new_full((batch_size, num_unique), -torch.inf)
        for idx in range(num_unique):
            mask = inverse == idx
            log_p_v_unique[:, idx] = torch.logsumexp(flow_log_prob[:, mask], dim=1)

        return log_p_v_unique[:, inverse]

    def _kde_log_density_1d(self, values):
        """
        Gaussian KDE log-density estimate at sample locations.
        values: (N,)
        returns: (N,)
        """
        n = values.shape[0]
        if n < 2:
            return torch.zeros_like(values)

        std = values.std(unbiased=False).clamp(min=1e-6)
        bandwidth = (1.06 * std * (float(n) ** (-1.0 / 5.0))).clamp(min=1e-6)

        diffs = (values[:, None] - values[None, :]) / bandwidth
        log_norm = torch.log(torch.tensor(math.sqrt(2.0 * math.pi), device=values.device, dtype=values.dtype) * bandwidth)
        log_kernel = -0.5 * diffs.pow(2) - log_norm
        return torch.logsumexp(log_kernel, dim=1) - torch.log(torch.tensor(float(n), device=values.device, dtype=values.dtype))

    def setup_flows(self):
        '''
        Set up normalizing flows for density estimation
        '''
        # Define flows
        num_layers = config.flow['num_layers']
        hidden_units = 64
        num_blocks = 2
        context_size = 1024
        
        # Set base distribution
        self.base = ConditionalDiagonalNormal(shape=[self.nfeatures], 
                                              context_encoder=MLP([context_size, 128, 64, self.nfeatures*2]))

        transforms = []
        for i in range(num_layers):
            transforms.append(RandomPermutation(features=self.nfeatures))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.nfeatures, 
                                                                hidden_features=hidden_units, 
                                                                context_features=context_size))

        self.transform = CompositeTransform(transforms)

    def point_estimate(self, x, y):
        '''
        Get point estimate for given inputs
        '''
        x = self.img_net(x)
        y = self.spec_net(y)
        x = x.view(int(self.bs),-1)
        y = y.view(int(self.bs),-1)
        z = torch.cat((x, y), -1)
        z = self.fully_connected_layer(z)
        return z

    def estimate_log_prob(self, x, y, zz, batch_size, vcirc_mu=None):
        '''
        Estimate log probability density for given inputs and parameters.
        zz: candidate parameter vectors of shape (N, nfeatures), e.g. [g1, g2, v_circ_norm]
        vcirc_mu: per-galaxy prior center, shape (batch_size,), units km/s (linear).
                  Required when mode == 2. Each galaxy's N candidates share the same mu.
        '''
        x = nn.functional.normalize(x, dim=[2,3])
        y = nn.functional.normalize(y, dim=[2,3])
        x = self.img_net(x)
        y = self.spec_net(y)
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)
        z = torch.cat((x, y), -1)
        # z = x
        z = self.layer_norm(z)
        num_candidates = zz.shape[0]
        z_rep = torch.repeat_interleave(z, repeats=num_candidates, dim=0)
        zz_rep = zz.repeat(batch_size, 1)
        flow_log_prob = self.flow.log_prob(zz_rep, context=z_rep).view(batch_size, -1)

        log_prob = flow_log_prob

        # mode 2 inference: replace flow marginal P_flow(v_circ|data) with TF prior P_TF(v_circ|data)
        # log p_new(g1,g2,v|data) = log p_flow(g1,g2,v|data) - log p_flow(v|data) + log p_TF(v|data)
        if self.mode == 2:
            assert vcirc_mu is not None, 'vcirc_mu (per-galaxy) must be provided for mode 2'

            v_norm_grid = zz[:, self.vcirc_idx].clamp(min=-1.0, max=1.0)
            flow_log_p_v = self._flow_v_marginal_from_grid(flow_log_prob, v_norm_grid)

            mu = vcirc_mu.to(device=log_prob.device, dtype=log_prob.dtype).reshape(-1, 1)
            assert mu.shape[0] == batch_size, 'vcirc_mu must have shape (batch_size,)'
            tf_log_p_v = self._tf_log_prob_from_vnorm(v_norm_grid.unsqueeze(0), mu)

            log_prob = flow_log_prob - flow_log_p_v + tf_log_p_v

        return log_prob

    def sample(self, x, y, num_samples, fp=None, vcirc_mu=None, return_log_prob=False):
        '''
        Sample from the conditional distribution p(params | x, y)
        num_samples: number of samples to draw per galaxy
        vcirc_mu: per-galaxy prior center, shape (1,), units km/s (linear).
                  Required when mode == 2. Each galaxy's N samples share the same mu.
        '''
        x = nn.functional.normalize(x, dim=[2,3])
        y = nn.functional.normalize(y, dim=[2,3])
        x = self.img_net(x)
        y = self.spec_net(y)
        x = x.view(1,-1)
        y = y.view(1,-1)
        z = torch.cat((x, y), -1)
        if fp is not None:
            fp = fp.view(fp.size(0), -1)  # Flatten fiber positions
            z = torch.cat((z, fp), dim=-1)
        z = self.layer_norm(z)

        # Sample from flow
        samples = self.flow.sample(num_samples, context=z)  # (1, num_samples, nfeatures)
        samples = samples.clamp(min=-1.5, max=1.5)

        if self.mode == 2:
            assert vcirc_mu is not None, 'vcirc_mu (per-galaxy) must be provided for mode 2'
            z_rep = torch.repeat_interleave(z, repeats=num_samples, dim=0)

            # Importance-resample flow samples so vcirc follows TF prior at inference time.
            candidates = samples[0]  # (num_samples, nfeatures)
            v_norm = candidates[:, self.vcirc_idx]
            v_circ = self._norm_to_vcirc(v_norm)

            mu = vcirc_mu.to(device=candidates.device, dtype=candidates.dtype).reshape(-1)
            assert mu.numel() == 1, 'vcirc_mu must have shape (1,) in sample()'
            tf_log_p_v = self._tf_log_prob_from_vnorm(v_norm, mu[0])
            flow_log_p_v = self._kde_log_density_1d(v_circ)

            log_w = tf_log_p_v - flow_log_p_v
            log_w = log_w - log_w.max()
            weights = torch.softmax(log_w, dim=0)
            resample_idx = torch.multinomial(weights, num_samples=num_samples, replacement=True)
            samples = candidates[resample_idx].unsqueeze(0)

        if return_log_prob:
            z_rep = torch.repeat_interleave(z, repeats=num_samples, dim=0)
            flow_log_prob = self.flow.log_prob(samples.view(num_samples, -1), context=z_rep)
            
            # For mode 2, compute TF-adjusted posterior log_prob
            if self.mode == 2:
                flow_log_prob = flow_log_prob + log_w[resample_idx]
            
            return samples.view(1, num_samples, -1), flow_log_prob
        return samples.view(1, num_samples, -1)

class MLP(nn.Module):
    '''
    A simple MLP with Linear and ReLU
    '''
    
    def __init__(self, layers):
        
        super(MLP,self).__init__()

        modules = nn.ModuleList([])
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2:
                modules.append(nn.ReLU(True))

        self.mlp = nn.Sequential(*modules)

    def forward(self,x):

        x = self.mlp(x)
        return x

class ResidualBlock(nn.Module):
    '''
    A residual block object that skips layers until stride > 1, i.e. the size of data shrinks
    '''
    
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        
        super(ResidualBlock,self).__init__()
        
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
            
    def forward(self,x):
        
        residual = x
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x += self.shortcut(residual)
        
        x = nn.ReLU(True)(x)
        return x

### ViT classes ###
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=6, img_size=48, embed_dim=512, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)  
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, img_size=48, patch_size=6, num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_embed = PatchEmbedding(in_channels, patch_size, img_size, embed_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return cls_token

### Spectra RNN ###
class SpecRNN(nn.Module):
    def __init__(self, nspec, hidden_size=256, num_layers=2, bidirectional=True):
        super().__init__()

        # Local feature extractor across time
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce temporal dim
        )

        # RNN across spectral dimension
        self.rnn_spec = nn.GRU(
            input_size=64,          # CNN feature dim per spectral bin
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat = self.cnn_spec(x) 
        
        feat = feat.mean(dim=-1)
        feat = feat.permute(0, 2, 1)

        # RNN along the spectral dimension
        rnn_out, _ = self.rnn_spec(feat)
        rnn_feat = rnn_out[:, -1, :]

        # Project to 512-dim feature
        out = self.proj(rnn_feat)
        return out
    
class LargeSpecRNN(nn.Module):
    def __init__(self, nspec, hidden_size=1024, num_layers=4, bidirectional=True):
        super().__init__()

        # Deeper local feature extractor across time
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce temporal dim
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Larger RNN across spectral dimension
        self.rnn_spec = nn.GRU(
            input_size=256,          # CNN feature dim per spectral bin
            hidden_size=hidden_size,  # Increased to 1024
            num_layers=num_layers,    # Increased to 4 layers
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0  # Add dropout between layers
        )

        # Larger output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

    def forward(self, x):
        # x: (batch, 1, nspec, time)
        batch_size = x.size(0)
        
        # CNN processing
        x = self.cnn_spec(x)  # (batch, 256, nspec, time/2)
        
        # Reshape for RNN: merge batch and time, treat spectral bins as sequence
        b, c, nspec, t = x.size()
        x = x.permute(0, 3, 2, 1)  # (batch, time, nspec, channels)
        x = x.reshape(b * t, nspec, c)  # (batch*time, nspec, channels)
        
        # RNN processing across spectral dimension
        x, _ = self.rnn_spec(x)  # (batch*time, nspec, hidden*2)
        
        # Global pooling across spectral dimension
        x = x.mean(dim=1)  # (batch*time, hidden*2)
        
        # Projection
        x = self.proj(x)  # (batch*time, 512)
        
        # Reshape back
        x = x.view(b, t, -1)  # (batch, time, 512)
        
        return x

### Spec CNN ###
class SpecCNN(nn.Module):

    def __init__(self, nspecs):
        super(SpecCNN, self).__init__()

        self.nspecs = nspecs

        self.cnn_spec = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512-2*nspecs, kernel_size=(self.nspecs, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512-2*nspecs),
            nn.ReLU(True),
            
        )

    def forward(self, x):
        
        x = self.cnn_spec(x)
        
        return x

### Image CNN ###
class ImgCNN(nn.Module):
    def __init__(self):
        super(ImgCNN, self).__init__()

        self.cnn_img = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, 2),
            
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256, 2),
            
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512, 2),
            
            nn.AvgPool2d(3),
            
        )

    def forward(self, x):
        
        x = self.cnn_img(x)
        
        return x


class DeconvNN(nn.Module):
    '''
    A deconv model in testing
    '''
    def __init__(self, batch_size, GPUs=1, 
                 nspec=config.data['nspec'], 
                 nfeatures=config.train['feature_number']):
        
        self.nfeatures = nfeatures
        self.bs = batch_size
        self.GPUs = GPUs
        
        super(DeconvNN, self).__init__()
        
        self.dnn_img = nn.Sequential(
            
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )
        
        ### Fully-connected layers
        self.linear = nn.Sequential(
            
            nn.Linear(self.nfeatures, 32),
            nn.Linear(32, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        )

    
    def forward(self, x):
        
        x = self.linear(x)
        
        x = x.view(int(self.bs),-1, 1, 1)
        
        x = self.dnn_img(x)
        
        return x
