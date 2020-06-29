"""Assortment of layers for use in QANet_models.py.

Author:
    Mengyao Bao(mb7570@nyu.edu)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class DepthwiseSeperableConv(nn.Module):
    """
    https://arxiv.org/pdf/1704.04861.pdf
    https://yinguobing.com/separable-convolution/
    """
    def __init__(self, in_channels, out_channels=128, kernel=7):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel,
                                   groups=in_channels, padding=(kernel-1)//2)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        return F.relu(x)                                                # x: [batch_size, seq_len, out_channels]


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):                          # q, k, v: [batch_size, seq_len, d_model]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],                        # pe broadcast to every sample in dim=0
                         requires_grad=False)
        return self.dropout(x)


class QAEncoderBlock(nn.Module):
    def __init__(self, kernel, d_model, d_ff, num_layers=4, n_head=8, dropout_rate=0.4):
        super(QAEncoderBlock, self).__init__()
        self.num_layers = num_layers
        assert d_model % n_head == 0
        self.d_k = d_model//n_head
        self.d_v = d_model//n_head

        self.position = PositionalEncoding(d_model, dropout=dropout_rate)
        self.cnns = clones(DepthwiseSeperableConv(in_channels=d_model, out_channels=d_model, kernel=kernel), num_layers)
        self.sublayer = clones(SublayerConnection(d_model, dropout_rate), num_layers+1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(n_head, d_model, self.d_k, self.d_v,)
        self.ff = PositionwiseFeedForward(d_model, d_ff)

    # TODO: mask
    def forward(self, x, mask):                             # x from ChaEmbedding: [batch_size, seq_len, hidden_size(d_model)]
        x = self.position(x)                                # x: [batch_size, seq_len, d_model]

        for i, cnn in enumerate(self.cnns):
            x = self.sublayer[i](x, cnn)                    # x: [batch_size, seq_len, d_model]

        q, attn = self.attention(x, x, x)                   # q: [batch_size, seq_len, d_model]
        x = self.layer_norm(x+q)

        return self.sublayer[-1](x, self.ff)                # output: [batch_size, seq_len, d_ff]


class QAOutput(nn.Module):
    def __init__(self, hidden_size):
        super(QAOutput, self).__init__()
        self.linear_1 = nn.Linear(hidden_size*2, 1)
        self.linear_2 = nn.Linear(hidden_size*2, 1)


    def forward(self, M0, M1, M2, mask):
        # Shapes: (batch_size, seq_len, 1)
        X1 = torch.cat([M0, M1], dim=2)
        X2 = torch.cat([M1, M2], dim=2)
        logits_1 = self.linear_1(X1)
        logits_2 = self.linear_1(X2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
