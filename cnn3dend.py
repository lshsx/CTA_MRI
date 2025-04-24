import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


class h_sigmoid(nn.Module):
	def __init__(self, inplace=True):
		super(h_sigmoid, self).__init__()
		self.relu = nn.ReLU6(inplace=inplace)

	def forward(self, x):
		return self.relu(x + 3) / 6


class h_swish(nn.Module):
	def __init__(self, inplace=True):
		super(h_swish, self).__init__()
		self.sigmoid = h_sigmoid(inplace=inplace)

	def forward(self, x):
		return x * self.sigmoid(x)


class CoordAtt(nn.Module):
	def __init__(self, inp, oup, reduction=32):
		super(CoordAtt, self).__init__()
		self.pool_d = nn.AdaptiveAvgPool3d((None, None, 1))
		self.pool_h = nn.AdaptiveAvgPool3d((None, 1, None))
		self.pool_w = nn.AdaptiveAvgPool3d((1, None, None))

		mip = max(8, inp // reduction)

		self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm3d(mip)
		self.act = h_swish()

		self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
		self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		identity = x

		n, c, d, h, w = x.size()
		x_d = self.pool_d(x).squeeze(2)
		x_h = self.pool_h(x).squeeze(3)
		x_w = self.pool_w(x).permute(0, 1, 4, 3, 2).squeeze(2)

		# 确保所有张量在第4个维度上具有相同的大小
		assert x_d.size(3) == x_h.size(3) == x_w.size(3)

		y = torch.cat([x_d, x_h, x_w], dim=1)

		y = self.conv1(y)
		y = self.bn1(y)
		y = self.act(y)

		x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)

		a_h = self.conv_h(x_h).sigmoid()
		a_w = self.conv_w(x_w).sigmoid()

		out = identity * a_w * a_h

		return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(x).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(x).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class DecoderSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(DecoderSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x, encoder_out):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class SELayer(nn.Module):
	def __init__(self, channel, reduction=1):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1, 1)
		return x * y.expand_as(x)


class BasicConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
	             bn=True, bias=False):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
		                      dilation=dilation, groups=groups, bias=bias)
		self.bn = nn.BatchNorm3d(out_planes, momentum=0.01, affine=True) if bn else None
		self.relu = nn.ReLU() if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class ChannelPool(nn.Module):
	def forward(self, x):
		return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 3
		self.compress = ChannelPool()
		self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = torch.sigmoid(x_out)
		self.attention = scale
		return x * self.attention


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        if x.dim() == 4:
            # 2D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), self.d_model)
            pe.requires_grad = False
            pos = torch.arange(0, x.size(2), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, 1::2] = torch.cos(pos * div_term)
        elif x.dim() == 5:
            # 3D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4),self.d_model)
            pe.requires_grad = False
            #print(pe.shape)
            pos = torch.arange(0, self.d_model/2, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            #print(pos.shape)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, :, 1::2] = torch.cos(pos * div_term)
        else:
            raise ValueError("Positional encoding input must have 4 or 5 dimensions")

        # 将 pe 移动到与 x 相同的设备上
        pe = pe.permute(0, 4, 1, 2, 3)
        pe = pe.to(x.device)
        return x + pe


class OrdinalEmbedding(nn.Module):
	def __init__(self, input_dim, embedding_dim):
		super(OrdinalEmbedding, self).__init__()
		self.fc = nn.Linear(input_dim, embedding_dim)

	def forward(self, x):
		return self.fc(x)


class MultiHeadAttention2D(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear transformations for Q, K, and V
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # Linear transformation for output
        self.W_out = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear transformation
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Splitting into multiple heads
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(0, 1).contiguous().view(batch_size, -1)
        output = self.W_out(attention_output)
        return output


class CognitiveAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(CognitiveAttentionModule, self).__init__()
        self.multihead_self_attention = MultiHeadAttention2D(input_dim, num_heads)
        self.multihead_cross_attention = MultiHeadAttention2D(input_dim, num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_q, input_k, input_v):
        # Self-attention
        self_attention_output = self.multihead_self_attention(input_q, input_q, input_q)

        # Cross-attention
        cross_attention_output = self.multihead_cross_attention(self_attention_output, input_k, input_v)

        # Layer normalization
        output = self.layer_norm(cross_attention_output + input_q)

        return output



class feature_Net_tp4(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor1 = nn.Sequential(
			nn.Conv3d(1, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
			nn.BatchNorm3d(16),
            nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
            nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
			nn.BatchNorm3d(64),
            nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(), 
			nn.Conv3d(128, 128, 1),
			nn.BatchNorm3d(128),
			nn.ReLU(),            
		)
		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
		)
		self.classifier2 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
		)
		self.classifier3 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
		)

		self.locat = PositionalEncoding3D(128)
		self.ca = SelfAttention(128)
		self.de = DecoderSelfAttention(128)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
	def forward(self, x):
		features_x = self.feature_extractor1(x)
		features_x = self.locat(features_x)
		map = self.ca(features_x)
		features_x = self.de(features_x, map) * features_x
		features_x = self.pool(features_x)
		features_x = features_x.view(features_x.shape[0], -1)
		logits1 = self.classifier1(features_x)
		logits2 = self.classifier2(features_x)
		logits3 = self.classifier3(features_x)
		return logits1,logits2,logits3,features_x



class Net_raw(nn.Module):
	def __init__(self, dim=256, dropout=0.5):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv3d(1, dim // 8, kernel_size=(3, 3, 3), padding=1),
			nn.BatchNorm3d(dim // 8),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv3d(dim // 8, dim // 4, kernel_size=(3, 3, 3), padding=1),
			nn.BatchNorm3d(dim // 4),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
			nn.BatchNorm3d(dim // 2),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2)
		)
		self.conv4 = nn.Sequential(
			nn.Conv3d(dim // 2, dim, kernel_size=(3, 3, 3), padding=1),
			nn.BatchNorm3d(dim),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2)
		)
		self.conv5 = nn.Sequential(
			nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1),
			nn.BatchNorm3d(dim),
			nn.ReLU(),
			nn.AdaptiveAvgPool3d((1, 1, 1))
		)
		self.ca=SelfAttention(256)
		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(dim, dim // 2),
			nn.ReLU(),
			nn.Linear(dim // 2, 1),
		)
		self.classifier2 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(dim, dim // 2),
			nn.ReLU(),
			nn.Linear(dim // 2, 1),
		)
		self.classifier3 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(dim, dim // 2),
			nn.ReLU(),
			nn.Linear(dim // 2, 1),
		)

	def forward(self, x):
		conv1_out = self.conv1(x)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		conv4_out = self.conv4(conv3_out)
		conv5_out = self.conv5(conv4_out)
		conv5_out=self.ca(conv5_out)*conv5_out
		out = conv5_out.view(conv5_out.size(0), -1)
		logits1 = self.classifier1(out)
		logits2 = self.classifier2(out)
		logits3 = self.classifier3(out)
		return logits1, logits2, logits3

