"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    #使用(1,) * (x.ndim - 1)可以让我们在不知道所有维度大小的情况下，仍然能够构造出正确的新形状的元组。
    #(1,) * (x.ndim - 1)创建了一个长度为x.ndim-1的元组，该元组中每个元素都是1。这个元组的目的是在新形状的第二个维度和以后的维度中插入1。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    #标量+张量，这里会自动调用广播机制，补全。
    #这里使得保留率+随机张量，就是引入保留率，例如保留率0.8，随机张量[0,1]，那么最终的区间就是[0.8,1.8]
    #torch.rand()是从[0,1]均匀分布中随机抽取数据，返回张量
    #torch.rand(2,1,1,1)生成的就是2*1*1*1只有batch维度的两个张量
    random_tensor.floor_()  # binarize
    #向下取整是drop_path失活的关键，0.7，0.8都会取整为0
    output = x.div(keep_prob) * random_tensor
    #这里关于keep_prob的理解：
    #为了保证训练和验证时候的期望一致，即随机变量的平均值一致。
    #输入张量除以keep_prob，拿0.2丢弃率为例，保留率为0.8，除以0.8就等于乘以1.2，就相当于丢弃20%路径，期望减小20%，留下的张量乘以1.2，期望再扩大20%，总体期望等于没变。
    #对输入张量进行缩放：x.div(keep_prob)。这一步的作用是将输入张量中每个元素除以keep_prob，以保持期望值不变。因为在进行dropout操作时，我们会将一部分元素设置为0，因此为了保持期望值不变，需要对剩余的元素进行缩放。
    #乘以随机张量：* random_tensor。这一步的作用是对缩放后的张量进行随机丢弃操作，即将一部分元素设置为0。由于random_tensor中每个元素的值为0或1，因此乘以random_tensor可以实现随机丢弃操作。
    #期望和均值都是描述一个随机变量的中心位置的指标，但是它们的计算方法略有不同。
    #均值是指随机变量所有取值的平均数，计算方法为将所有取值相加后再除以取值个数。
    #而期望则是指随机变量在一定条件下的平均值，计算方法为将每个取值乘以其出现的概率后相加。
    #在概率论和统计学中，期望是一个很重要的概念，它可以用来描述一个随机变量的分布情况，同时也可以用来计算随机变量的方差和其他统计量。
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        #这段代码是 PatchEmbed 层的前向传播函数。它的作用是将输入张量 x 转换为形状为 (batch_size, num_patches, embed_dim) 的输出张量。
        #具体来说，这段代码首先获取输入张量的形状，并检查其高度和宽度是否与 PatchEmbed 层的期望值相同。然后，它将输入张量 x 传递给 proj 层进行卷积操作，并将输出张量的最后两个维度（即高度和宽度）展平成一个维度。接着，它使用 transpose 函数将第二个和第三个维度交换，以便后面的全连接层可以更方便地处理输出张量。
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        #缩放因子
        #Attention(Q,K,V)=softmax((Q*K^T)/dk^-0.5)*V 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        #num_patches就是把一张224图片分割成了14*14的小块，num_patches=14
        #num_patched+1就是加了一个class token，类别标识，类别张量
        #total_embed_dim就是通过全连接层改变的通道的个数，升维
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 这里的 reshape 操作可以改变张量的形状是因为 x 张量的元素总数与 B * N * C 相同，因此可以通过改变张量的形状，将其重新排列成一个新的张量，而不改变张量的元素数量和元素值。
        # 具体来说，x 张量的形状为 (B, N, C)，其中 B 是 batch size，N 是序列长度，C 是每个 token 的特征维度。self.qkv(x) 返回的张量的形状为 (B, N, 3 * total_embed_dim)，
        # 其中 total_embed_dim 是经过线性变换后的特征维度。因此，reshape(B, N, 3, self.num_heads, C // self.num_heads) 将这个张量重新排列成一个新的张量，
        # 其中第三维被拆分成三个子维度，分别对应于查询、键和值。permute(2, 0, 3, 1, 4) 将新的张量按照指定的顺序重新排列，以便后续的计算。
        # 最终得到的张量形状为 (3, B, self.num_heads, N, C // self.num_heads)。
       
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 把3放在第一个维度，就对应的是qkv三个张量，通过索引分别取出qkv
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # k.transpose(-2, -1) 和 k 的转置是等价的，3*2变成2*3
        # @矩阵乘法的符号
        # ab * ba = aa
        # * self.scale 进行normal处理
        attn = attn.softmax(dim=-1)
        #dim=-1意思就是在张量的每一行上进行softmax处理
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    #Multi-Layer Perceptron
    #多层感知器
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        #GELU的最小值为-0.21，值域为[ − 0.21 , + ∞ ] 
        #Relu等分段线性激活函数并不光滑，而Gelu函数在靠近0的值具有连续性，近似等于x，光滑的激活函数有较好的泛化能力和稳定的优化能力,可以提高模型的性能
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
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
                 #层归一化
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        #depth这里指的就是重复堆叠encoder block的次数
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        # 在ViT中，self.num_tokens是一个类成员变量，用于指定输入图像的序列长度。在ViT中，输入图像被分成一系列的图像块（image patches），每个图像块都被视为一个令牌（token）。
        # 在Transformer的输入序列中，每个令牌都被表示为一个向量，这些向量被送入Transformer模型进行处理。
        # self.num_tokens的值取决于是否使用了蒸馏（distillation）技术。如果使用了蒸馏技术，则每个图像块都会被分成两个子块，每个子块都被视为一个令牌，因此self.num_tokens的值为2。
        # 如果没有使用蒸馏技术，则每个图像块被视为一个令牌，因此self.num_tokens的值为1。
        # 在ViT中，蒸馏（distillation）技术是一种模型压缩技术，用于将一个大型的模型压缩成一个小型的模型，同时尽可能地保留原始模型的性能。
        # 在蒸馏技术中，我们有两个模型：一个是大模型，也称为“教师模型”（teacher model），另一个是小模型，也称为“学生模型”（student model）。
        # 我们使用教师模型来生成训练数据，然后使用学生模型来学习这些训练数据。学生模型的目标是尽可能地模仿教师模型的输出，同时保持模型尺寸较小。
        # 在ViT中，蒸馏技术用于将大型的ViT模型压缩成一个小型的模型，以便在计算资源受限的情况下使用。在蒸馏过程中，每个输入图像块都会被分成两个子块，每个子块都被视为一个令牌。
        # 这些子块的特征表示被送入教师模型进行处理，然后将这些特征表示的平均值作为每个输入图像块的特征表示。学生模型使用这些平均特征表示来学习输入图像的表示。
        # 因此，在使用蒸馏技术时，输入图像被视为由两个令牌组成的序列，而不是一个令牌序列。
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # 在ViT模型中，representation_size指定了表示层的大小，即将输入特征向量（patch embeddings）映射到一个固定大小的向量表示。
        # 这个向量表示可以看作是输入图像的全局特征，它会被送入模型的后续层中进行分类或回归等任务。如果representation_size被指定为None，
        # 则表示模型不使用表示层，而是直接使用输入特征向量进行分类或回归等任务。
        # 如果指定了representation_size，则会使用一个包含一个全连接层和一个激活函数的序列来将输入特征向量映射到指定大小的表示层上。
        # 在ViT模型中，如果指定了表示层的大小，则使用一个包含一个全连接层和一个激活函数的序列来将输入特征向量映射到指定大小的表示层上，
        # 这个序列被称为pre_logits模块。pre_logits模块中包含一个全连接层和一个tanh激活函数，用于将输入特征向量映射到指定大小的表示层上。如果没有指定表示层大小，
        # 则使用一个恒等函数来保持输入特征向量不变。pre_logits模块的作用是对输入特征向量进行一定的变换，从而得到更具有代表性的特征向量，以提高模型的性能。
        if representation_size and not distilled:
            #判断两个都为真值才能运行下面的代码
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
