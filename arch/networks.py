import torch
from torch import nn
import normflows as nf

import config

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

### Main Network ###
class ForkCNN(nn.Module):
    '''
    CNN used for direct prediction of parameters.
    '''
    def __init__(self, 
                 mode=0,    # 0 = point estimate, 1 = density estimate
                 base=None,
                 flows=None,
                 batch_size=config.train['batch_size'],
                 nfeatures=config.train['feature_number'],
                 nspec=config.data['nspec']):

        self.bs = batch_size
        self.nfeatures = nfeatures
        self.nspecs = nspec
        if mode == 0 or mode == 1:
            self.mode = mode
        else:
            raise ValueError('Mode can only be 0 (point estimate) or 1 (density estimate)!')
        if mode == 1 and (base is None or flows is None):
            raise Exception('Need to specify base and flows if doing density estimate!')

        super(ForkCNN, self).__init__()
        
        # self.cnn_img = nn.Sequential(
            
        #     nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
            
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
            
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     ResidualBlock(64, 128),
        #     ResidualBlock(128, 128),
        #     ResidualBlock(128, 128),
        #     ResidualBlock(128, 128, 2),
            
        #     ResidualBlock(128, 256),
        #     ResidualBlock(256, 256),
        #     ResidualBlock(256, 256),
        #     ResidualBlock(256, 256),
        #     ResidualBlock(256, 256, 2),
            
        #     ResidualBlock(256, 512),
        #     ResidualBlock(512, 512),
        #     ResidualBlock(512, 512),
        #     ResidualBlock(512, 512),
        #     ResidualBlock(512, 512, 2),
            
        #     nn.AvgPool2d(3),
            
        # )
        
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
            
            nn.Conv2d(256, 512, kernel_size=(nspec, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
        )
        
        if self.mode == 0:
            ### Fully-connected layers
            self.fully_connected_layer = nn.Sequential(
                # make sure the first number is equal to the sum of final # of channels in both img and spec branches
                nn.Linear(1024, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, self.nfeatures)
            )
            self.loss = nn.MSELoss()
        elif self.mode == 1:
            # Normalizing flow for density estimation
            self.flow = nf.ConditionalNormalizingFlow(base, flows)
            with torch.no_grad():
                for param in self.flow.parameters():
                    param.zero_()  # Initialize flow to identity


        # Vision Transformer for image feature extraction
        self.vit = VisionTransformer(in_channels=1, 
                                     embed_dim=512, 
                                     img_size=48, 
                                     patch_size=6, 
                                     num_layers=6, 
                                     num_heads=8, 
                                     mlp_ratio=4.0, 
                                     dropout=0.1)

    
    def forward(self, x, y, true):
        
        # Feature extraction from img and spec
        # x = self.cnn_img(x)
        x = self.vit(x)
        y = self.cnn_spec(y)
        
        # Flatten and concatenate
        # x = x.view(int(self.bs),-1)
        y = y.view(int(self.bs),-1)
        z = torch.cat((x, y), -1)

        # Point/density estimate
        if self.mode == 0:
            z = self.fully_connected_layer(z)
            loss = self.loss(z, true)
        elif self.mode == 1:
            # For normalizing flows, directly output loss
            loss = self.flow.forward_kld(true, context=z)

        return loss

    def estimate_log_prob(self, x, y, zz):
        '''
        Estimate log probability density for given inputs and parameters
        '''
        x = self.vit(x)
        y = self.cnn_spec(y)
        y = y.view(1, -1)
        z = torch.cat((x, y), -1)
        z = z.repeat(zz.shape[0], 1)
        log_prob = self.flow.log_prob(zz, context=z)

        return log_prob


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

    
## NN calibration
class CaliNN(nn.Module):
    
    def __init__(self, nfeatures=config.cali['feature_number']):
        super(CaliNN, self).__init__()
        
        self.main_net = nn.Sequential(
            nn.Tanh(),
            nn.Linear(nfeatures,10),
            nn.Tanh(),
            nn.Linear(10,10),
            nn.Tanh(),
            nn.Linear(8,2),
        )

    
    def forward(self, x):
        
        x = self.main_net(x)
        return x
    
    


