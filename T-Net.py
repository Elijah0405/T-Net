import torch
import torch.nn as nn
def Conv1(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=2, padding=0),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )
def Conv2(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=0),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )
def Conv3(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=kernel_size//2),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )
def Conv4(input_channel, output_channel, kernel_size1, kernel_size2):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=(kernel_size1, kernel_size2), 
                  stride=1, padding=(kernel_size1//2, kernel_size2//2)),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )

# Backbone
class Backbone(nn.Module):
    def __init__(self, num_classes):
        super(Backbone, self).__init__()
        self.block1 = nn.Sequential(
            Conv1(3, 32, 3),
            Conv2(32, 32, 3),
            Conv3(32, 64, 3)
        )

        self.block2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.block2_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.block2_3 = Conv1(64, 96, 3)

        self.block3_1 = nn.Sequential(
            Conv3(224, 64, 1),
            Conv2(64, 96, 3)
        )
        self.block3_2 = nn.Sequential(
            Conv3(224, 64, 1),
            Conv4(64, 64, 7, 1),
            Conv4(64, 64, 1, 7),
            Conv2(64, 96, 3)
        )

        self.block4_1 = nn.Sequential(
            Conv1(192, 96, 3),
            Conv4(96, 96, 3, 1),
            Conv4(96, 96, 1, 3),
        )
        self.block4_2 = nn.Sequential(
            Conv3(192, 96, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.block4_3 = nn.Sequential(
            Conv3(192, 96, 1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.block4_4 = Conv1(192, 96, 3)

        self.block5 = Conv1(384, 512, 3)

        self.block6 = nn.MaxPool2d(kernel_size=12, stride=1, padding=0)

        self.cbam1 = CBAM(224)
        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        
        self.linear = nn.Linear(512,num_classes)
        self.cbam = CBAM(512)
        self.sa = SA(img_size=12,patch_size=1,in_c=512, num_classes=num_classes,
                                 embed_dim=512,depth=1,num_heads=8,drop_ratio=0.,
                                 attn_drop_ratio=0.,drop_path_ratio=0.)

        self.weight1 = nn.Parameter(torch.Tensor([0.2]))
        self.weight2 = nn.Parameter(torch.Tensor([0.3]))
        self.weight3 = nn.Parameter(torch.Tensor([0.5]))
        #nn.init.trunc_normal_(self.weight1, std=0.02)
        #nn.init.trunc_normal_(self.weight2, std=0.02)
        #nn.init.trunc_normal_(self.weight3, std=0.02)


    def forward(self, x):
        x = self.block1(x)
        x1 = self.block2_1(x)
        x2 = self.block2_2(x)
        x3 = self.block2_3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.cbam1(x)
        x1 = self.block3_1(x)
        x2 = self.block3_2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.cbam2(x)
        x1 = self.block4_1(x)
        x2 = self.block4_2(x)
        x3 = self.block4_3(x)
        x4 = self.block4_4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.cbam3(x)
        x = self.block5(x)
        #512

        x_sa = self.sa(x)
        #[B,512]
        out1 = self.linear(x_sa)

        x_cbam = self.cbam(x)
        x_cbam = self.block6(x_cbam)
        x_cbam = x_cbam.view(x_cbam.size(0), -1)
        #[B,512]
        #x_cbam.size(0)保持第一维度大小不变，展平后面的维度
        out2 = self.linear(x_cbam)
        
        x_normal = self.block6(x)
        x_normal = x_normal.view(x.size(0), -1)   
        #[B,512]
        out3 = self.linear(x_normal)
        
        #out = torch.cat([x_sa, x_cbam, x_normal], dim=1)
        #out = self.weight1 * x_sa + self.weight2 * x_cbam + self.weight3 * x_normal
        out = self.weight1 * out1 + self.weight2 * out2 + self.weight3 * out3
        #out = self.linear(out)
        #512 --> 6

        return out
    
# T-Net
class TNet(nn.Module):
    def __init__(self, num_classes=6):
        super(TNet, self).__init__()
        
        self.block = Backbone(num_classes=num_classes)

    def forward(self, x):
        x = self.block(x)
        return x


class ChannelAttention(nn.Module):  # Channel attention module
    def __init__(self, channels, ratio=16):  # r: reduction ratio=16
        super(ChannelAttention, self).__init__()

        hidden_channels = channels // ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # global avg pool
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # global max pool
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False), 
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        return self.sigmoid(
            self.mlp(x_avg) + self.mlp(x_max)
        )


class SpatialAttention(nn.Module):  # Spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  
        x_max = torch.max(x, dim=1, keepdim=True)[0]  
        return self.sigmoid(
            self.conv(torch.cat([x_avg, x_max],dim=1))
        )


class CBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, channels, ratio=16):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention = SpatialAttention()  # Spatial attention module

    def forward(self, x):
        f1 = self.channel_attention(x) * x  
        f2 = self.spatial_attention(f1) * f1  
        return f2
    
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.3):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = keep_prob + random
        random_tensor = random_tensor.floor_()
        output = x / keep_prob * random_tensor
        return output

class PatchEmbed(nn.Module):
    def __init__(self, img_size=14 ,patch_size=1, in_c=768, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.proj = nn.Conv2d(768, 768, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MHSA(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=12,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.2,
                 proj_drop_ratio=0.2):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.2,
                 attn_drop_ratio=0.2,
                 drop_path_ratio=0.2):
        super(Block, self).__init__()
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x
    
class transfomer(nn.Module):
    def __init__(self, img_size=14, patch_size=1, in_c=768, num_classes=6,
                 embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.2,attn_drop_ratio=0.2, drop_path_ratio=0.2, 
                 embed_layer=PatchEmbed):
        super(transfomer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim 
        self.num_tokens = 1

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)
            for i in range(depth)
        ])

        self.pre_logits = nn.Identity()
        # Classifier head(s)
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    
        # Weight init
        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        #nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.pre_logits(x[:, 0])
        # here we need to make sure one thing, why do we get a two dimensions torch when a three dimension torch pass through x[:, 0]?
        # Meanwhile this new torch keep the same first dimenson and third dimension to the old one.
        # Through experiment, we find that x[:, 0] is equal to x[:, 0, :], the last slice operation is omitted.

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x

class SA(nn.Module):
# self attention classification
    def __init__(self, in_c, img_size=14, patch_size=1, num_classes=6, 
                 embed_dim=768, depth=1, num_heads=12,drop_ratio=0.1, 
                 attn_drop_ratio=0.1, drop_path_ratio=0.1):
        super(SA, self).__init__()
        #self.channel_attention = ChannelAttention(channels=in_c)  # Channel attention module
        #self.spatial_attention = SpatialAttention()  # Spatial attention module
        self.transfomer = transfomer(img_size=img_size,
                          patch_size=patch_size,
                          in_c=in_c,
                          num_classes=num_classes,
                          embed_dim=embed_dim,
                          depth=depth,
                          num_heads=num_heads,
                          drop_ratio=drop_ratio,
                          attn_drop_ratio=attn_drop_ratio,
                          drop_path_ratio=drop_path_ratio)

    def forward(self, x):
        x = self.transfomer(x)
        return x
    


if __name__=="__main__":
    a=torch.randn(2,3,224,224)
    net=TNet()
    #print(net)
    print(net(a).shape)