import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet
from torchinfo import summary
from collections import OrderedDict

EPS = 1e-7


class UNet(nn.Module):
    def __init__(self, cout):
        super(UNet, self).__init__()
        print('Building Encoder, Decoder ...')
        self.encoder = Encoder()
        self.decoder = Decoder(cout=cout)
        print('Finished building model')

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(x, features)
        return out


class Encoder(nn.Module):
    def __init__(self, basemodel_name='tf_efficientnet_b5_ap', pretrained=True):
        super(Encoder, self).__init__()
        print(f'Loading base model {basemodel_name} ...')
        torch.hub.set_dir('./.cache')
        self.basemodel = getattr(geffnet, basemodel_name)(pretrained=pretrained)
        # basemodel = geffnet.tf_efficientnet_b5_ap(pretrained=pretrained)
        self.basemodel.global_pool = nn.Identity()
        self.basemodel.classifier = nn.Identity()
        print(f'Finished loading base model {basemodel_name}')

    def forward(self, x):
        features = [x]
        for k, v in self.basemodel._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
                    # f_dict['block'+ki] = features[-1]
            else:
                features.append(v(features[-1]))
                # f_dict[k] = features[-1]
        '''last_feature = self.basemodel(x)  # 为了训练预训练模型
        last_feature = self.un_flatten(last_feature)'''
        return features


class Decoder(nn.Module):
    def __init__(self, cout, bottleneck=2048, feature_dim=2048):
        super(Decoder, self).__init__()
        self.up_block0 = nn.Sequential(
            nn.Conv2d(bottleneck, feature_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        self.up_block1 = WeightedDecoderBlock(feature_dim, feature_dim//2, [1, 1], 176)
        self.up_block2 = WeightedDecoderBlock(feature_dim//2, feature_dim//4, [2, 2], 64)
        self.up_block3 = WeightedDecoderBlock(feature_dim//4, feature_dim//8, [4, 4], 40)
        self.up_block4 = WeightedDecoderBlock(feature_dim//8, feature_dim//16, [8, 8], 24)
        self.out_block = nn.Sequential(
            nn.Conv2d(feature_dim//16, feature_dim//16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim//16),
            nn.GELU(),
            nn.Conv2d(feature_dim//16, feature_dim//16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim//16),
            nn.GELU(),
            nn.Conv2d(feature_dim//16, cout, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.out_block = OutBlock(in_channels=features//16+24, mid_channels=features//32, parallel_channels=24, cout=cout)
        # 24 is the channel number of parallel block

    def forward(self, x, features):
        feature1, feature2, feature3, feature4, feature5 = [features[4], features[5],
                                                            features[6], features[8], features[13]]
        '''feature1, feature2, feature3, feature4 = [features[4], features[5], features[6], features[8]]
        feature5 = last_feature.detach()'''
        # [torch.Size([2, 24, 240, 320]), torch.Size([2, 40, 120, 160]), torch.Size([2, 64, 60, 80]), torch.Size([2, 176, 30, 40]), torch.Size([2, 2048, 15, 20])]

        up0 = self.up_block0(feature5)  # [2048, 15, 20]
        up1 = self.up_block1(up0, feature4)  # [1024, 30, 40]
        up2 = self.up_block2(up1, feature3)  # [512, 60, 80]
        up3 = self.up_block3(up2, feature2)  # [256, 120, 160]
        up4 = self.up_block4(up3, feature1)  # [128, 240, 320]

        out = self.out_block(up4)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class WeightedDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size, feature_dim=0):
        super(WeightedDecoderBlock, self).__init__()
        # dilation=1,3,5,7 of encoded feature
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim+feature_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.pwt = PatchwiseAttention(out_dim, out_dim, patch_size)

    def forward(self, x, feature):
        # upsample coming up feature to match encoded feature
        up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if feature is not None:
            c1 = self.conv1(torch.cat([up, feature], 1))
        else:
            c1 = self.conv1(up)
        attention_map = self.pwt(c1)
        return c1*attention_map + self.conv2(c1)


class PatchwiseAttention(nn.Module):
    def __init__(self, feature_dim=128, embedding_dim=128, patch_size=[16, 16], num_heads=4, dim_feedforward=1024):
        super(PatchwiseAttention, self).__init__()
        self.conv_embedding = nn.Conv2d(feature_dim, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        # self.pos_embedding = nn.Parameter(torch.rand(884, embedding_dim), requires_grad=True)

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, activation=F.gelu)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

    def dynamic(self, name: str, module_class, *args, **kwargs):
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)

    def forward(self, feature):
        conv_embeddings = self.conv_embedding(feature)  # Batch_size,Embedding_dim,Patch_size[0]*[1]
        b, c, p_h, p_w = conv_embeddings.shape
        conv_embeddings = conv_embeddings.flatten(2)

        embeddings = conv_embeddings

        # change to Patch_size,Batch_size,Embedding_dim format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        # self-attentioned feature
        attention_map = self.transformer_encoder(embeddings)
        attention_map = attention_map.permute(1, 2, 0)
        # attention_map = self.dynamic("linear", LinearRegressor, in_dim=p_h*p_w, hidden_dim=512, out_dim=p_h*p_w)(attention_map)
        attention_map = attention_map.view(b, c, p_h, p_w)
        attention_map = F.interpolate(attention_map, size=[feature.shape[2], feature.shape[3]], mode="area")  # weight the original feature by attention map of patch-size wise
        return attention_map


if __name__ == '__main__':
    x = torch.rand(1, 3, 416, 544)
    # x = torch.rand(1, 3, 480, 640)
    model = UNet(cout=1)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = model(x)
    summary(model, input_data=x)
    # print(pytorch_total_params)
