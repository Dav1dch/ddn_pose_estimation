import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.models.superpoint import SuperPoint
import torchvision.transforms as transforms


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, x2):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        q, _, _ = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        _, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv2)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNormAttn(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, x2):
        for attn, ff in self.layers:
            x = attn(x, x2) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 7),
        )

    def forward(self, img1, img2):
        x = self.to_patch_embedding(
            img1
        )  # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        x2 = self.to_patch_embedding(img2)
        b, n, _ = x.shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(
            self.cls_token, "() n d -> b n d", b=b
        )  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # 将cls_token拼接到patch token中去       (b, 65, dim)

        x2 = torch.cat((cls_tokens, x2), dim=1)
        x += self.pos_embedding[:, : (n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        x2 += self.pos_embedding[:, : (n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)
        x2 = self.dropout(x2)

        x = self.transformer(x, x2)  # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  # (b, dim)

        x = self.to_latent(x)  # Identity (b, dim)
        # print(x.shape)

        return self.mlp_head(x)  #  (b, num_classes)


class ViTSP(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )
        self.sp_to_patch_embedding = nn.Sequential(
            Rearrange(" b n c h w -> b n (h w c)"),
            nn.Linear(768, dim),
        )
        config = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": 64,
            },
        }
        self.transform = transforms.Grayscale()

        self.extraction = SuperPoint(config["superpoint"])
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.sp_pos_embedding = nn.Linear(2, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 7),
        )

    def forward(self, img1, img2):
        img1Gray = self.transform(img1) / 255
        img2Gray = self.transform(img2) / 255
        extract1 = torch.stack(self.extraction({"image": img1Gray})["keypoints"]).cuda()
        extract2 = torch.stack(self.extraction({"image": img2Gray})["keypoints"]).cuda()
        x_ = []
        for i in range(extract1.shape[0]):
            tmp = []
            for y, x in extract1[i]:
                y, x = int(y), int(x)
                tmp.append(img1[i, :, x - 8 : x + 8, y - 8 : y + 8])
            x_.append(torch.stack(tmp))
        x_1 = torch.stack(x_)
        x_ = []
        for i in range(extract2.shape[0]):
            tmp = []
            for y, x in extract2[i]:
                y, x = int(y), int(x)
                tmp.append(img2[i, :, x - 8 : x + 8, y - 8 : y + 8])
            x_.append(torch.stack(tmp))
        x_2 = torch.stack(x_)
        x = self.sp_to_patch_embedding(x_1)
        x2 = self.sp_to_patch_embedding(x_2)
        pos_embedding1 = self.sp_pos_embedding(extract1)
        pos_embedding2 = self.sp_pos_embedding(extract2)

        """

        x = self.to_patch_embedding(
            img1
        )  # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim

        x2 = self.to_patch_embedding(img2)
        b, n, _ = x.shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(
            self.cls_token, "() n d -> b n d", b=b
        )  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # 将cls_token拼接到patch token中去       (b, 65, dim)

        x2 = torch.cat((cls_tokens, x2), dim=1)
        x += self.pos_embedding[:, : (n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        x2 += self.pos_embedding[:, : (n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        """
        x += pos_embedding1
        x2 += pos_embedding2
        x = self.dropout(x)
        x2 = self.dropout(x2)

        x = self.transformer(x, x2)  # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  # (b, dim)

        x = self.to_latent(x)  # Identity (b, dim)
        # print(x.shape)

        return self.mlp_head(x)  #  (b, num_classes)


def test():
    model = ViT(
        image_size=(480, 640),
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    ).cuda()

    img = torch.randn(16, 3, 480, 640).cuda()
    img2 = torch.randn(16, 3, 480, 640).cuda()
    preds = model(img, img2)
    print(preds.shape)
    return


if __name__ == "__main__":
    test()
