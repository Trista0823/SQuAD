"""Top-level QANet model class.

Author:
    Mengyao Bao(mb7570@nyu.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from BiDAF_layers import CharEmbeddings, BiDAFAttention
from QANet_layers import clones, QAEncoderBlock, QAOutput, DepthwiseSeperableConv


class QANet(nn.Module):
    """QANet model for SQuAD.

    Based on the paper:
    "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
    by Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, Quoc V. Le
    (https://arxiv.org/pdf/1804.09541.pdf).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices(lookup) to get word vectors and embed character indices(fine tune) with
                           Conv1d, then go throw a Highway and dropout.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        kernel: size of the convolving kernel
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, kernel):
        super(QANet, self).__init__()
        # Embed layer
        self.embedding = CharEmbeddings(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        hidden_size=hidden_size,            # hidden_size = 128
                                        dropout_rate=drop_prob)

        # Encoder layer
        self.query_encoders = QAEncoderBlock(kernel=kernel, d_model=hidden_size, d_ff=hidden_size, num_layers=4)
        self.context_encoders = QAEncoderBlock(kernel=kernel, d_model=hidden_size, d_ff=hidden_size, num_layers=4)
        # Attention layer
        self.att = BiDAFAttention(hidden_size=hidden_size)
        # Model layer
        self.proj = DepthwiseSeperableConv(in_channels=hidden_size*4, out_channels=hidden_size)
        self.models = clones(QAEncoderBlock(kernel=kernel, d_model=hidden_size,
                                            d_ff=hidden_size, num_layers=2), 2)
        # Output layer
        self.output = QAOutput(hidden_size=hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        """
        Args:
            cw_idxs (torch.tensor): context consists of word indexes, shape: [batch_size(max 400), seq_len]
            qw_idxs (torch.tensor): query consists of word indexes, shape: [batch_size(max 50), seq_len]
            cc_idxs (torch.tensor): context consists of char indexes, shape: [batch_size(max 400), seq_len, word_len(16)]
            qc_idxs (torch.tensor): query consists of char indexes, shape: [batch_size(max 50), seq_len, word_len(16)]

        Returns:
            out (tuple of torch.tensor): logits of start and end position, each of shape [batch_size, seq_len]
        """

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs       # c_mask: [batch_size, seq_len]
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs       # q_mask: [batch_size, seq_len]

        q_embed = self.embedding(qw_idxs, qc_idxs)          # q_embed: [batch_size, seq_len, hidden_size]
        c_embed = self.embedding(cw_idxs, cc_idxs)          # c_embed: [batch_size, seq_len, hidden_size]

        q_enc = self.query_encoders(q_embed, q_mask)        # q_enc: [batch_size, seq_len, hidden_size]
        c_enc = self.context_encoders(c_embed, c_mask)      # c_enc: [batch_size, seq_len, hidden_size]

        att = self.att(c_enc, q_enc, c_mask, q_mask)        # att: [batch_size, seq_len, hidden_size*4]
        att = att.transpose(1, 2)                           # att: [batch_size, hidden_size*4, seq_len]

        M1 = self.proj(att).transpose(1, 2)                 # M1, M2, M3: [batch_size, seq_len, hidden_size*4]

        for model in self.models:
            M1 = model(M1, c_mask)

        M2 = M1
        for model in self.models:
            M2 = model(M2, c_mask)

        M3 = M2
        for model in self.models:
            M3 = model(M3, c_mask)
        out = self.output(M1, M2, M3, c_mask)               # output: (log_p1, log_p2), each of shape: [batch_size, seq_len]
        return out



