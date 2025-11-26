import torch
import torch.nn.functional as F
# from scaled_dot_product_attention import scaled_dot_product_attention
import math


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    # 确保所有张量在同一设备
    device = query.device  # 获取query的设备

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale


    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)

    if is_causal:
        assert attn_mask is None
        # 生成掩码时对齐设备
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)  # 修改点2：添加device参数
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)  # 确保数据类型一致

    if attn_mask is not None:
        # 将attn_mask移动到当前设备（若需要）
        attn_mask = attn_mask.to(device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 ffn=True,
                 ffn_dim_expansion=1
                 ):
        super(TransformerLayer, self).__init__()

        # multi-head attention
        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.v_proj = torch.nn.Linear(feature_dim, feature_dim)

        self.merge = torch.nn.Linear(feature_dim, feature_dim)

        # self.multi_head_attn = torch.nn.MultiheadAttention(feature_dim, 2, batch_first=True, device='cuda')

        self.norm1 = torch.nn.LayerNorm(feature_dim)

        self.ffn = ffn

        if self.ffn:
            in_channels = feature_dim * 2
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(in_channels * ffn_dim_expansion, feature_dim, bias=False),
            )

            self.norm2 = torch.nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        message = scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        message = self.merge(message)

        # message, _ = self.multi_head_attn(query, key, value, need_weights=False)
        message = self.norm1(message)

        if self.ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class FeatureAttention(torch.nn.Module):
    def __init__(self, feature_dim, num_layers, ffn=True, ffn_dim_expansion=1, post_norm=False):
        super(FeatureAttention, self).__init__()

        self.layers = torch.nn.ModuleList([
            TransformerLayer(feature_dim, ffn=ffn, ffn_dim_expansion=ffn_dim_expansion
                             )
            for i in range(num_layers)])

        self.post_norm = post_norm

        if self.post_norm:
            self.norm = torch.nn.BatchNorm2d(feature_dim)

    def forward(self, concat_features0):

        b, c, h, w = concat_features0.shape

        concat_features0 = concat_features0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        # 交换了前一半batch和后一半batch的数据
        concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)
        # transformer模块的交叉注意力

        # feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
        for layer in self.layers:
            concat_features0 = layer(concat_features0, concat_features1)
            concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # reshape back
        concat_features0 = concat_features0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        if self.post_norm:
            concat_features0 = self.norm(concat_features0)

        return concat_features0


class FlowAttention(torch.nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, feature_dim):
        super(FlowAttention, self).__init__()

        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, feature, flow):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        b, c, h, w = feature.size()

        feature = feature.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        flow = flow.flatten(-2).permute(0, 2, 1)

        query = self.q_proj(feature)  # [B, H*W, C]
        key = self.k_proj(feature)  # [B, H*W, C]

        flow = scaled_dot_product_attention(query, key, flow)

        flow = flow.view(b, h, w, 2).permute(0, 3, 1, 2)

        #
        # b, c, h, w = feature.size()
        #
        # query = feature.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        #
        # # author of GMFlow projected the `query` twice
        # # this doesn't affect the network since projection is a linear operation and can be merged
        # query = self.q_proj(query)  # [B, H*W, C]
        # key = self.k_proj(query)  # [B, H*W, C]
        #
        # value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]
        #
        # scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        # prob = torch.softmax(scores, dim=-1)
        #
        # out = torch.matmul(prob, value)  # [B, H*W, 2]
        # out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]
        return flow
