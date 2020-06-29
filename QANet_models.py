"""Top-level QANet model class.

Author:
    Mengyao Bao(mb7570@nyu.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from BiDAF_layers import CharEmbeddings, BiDAFAttention
from QANet_layers import clones, QAEncoderBlock, QAOutput


class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, dropout_rate, kernel):
        super(QANet, self).__init__()
        # Embed layer
        self.embedding = CharEmbeddings(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        hidden_size=hidden_size,            # hidden_size = 128
                                        dropout_rate=dropout_rate)
        # Encoder layer
        self.query_encoders = QAEncoderBlock(kernel=kernel, d_model=hidden_size, d_ff=hidden_size, num_layers=4)
        self.context_encoders = QAEncoderBlock(kernel=kernel, d_model=hidden_size, d_ff=hidden_size, num_layers=4)
        # Attention layer
        self.attn = BiDAFAttention(hidden_size=hidden_size)
        # Model layer
        # TODO share weights between each of the 3 repetitions of the model encoder?
        self.models = clones(QAEncoderBlock(kernel=kernel, d_model=hidden_size,
                                            d_ff=hidden_size, num_layers=2), 3)
        # Output layer
        self.output = QAOutput(hidden_size=hidden_size)

    def forward(self, word_idx, char_idx):
        # word_idx: [batch_size, seq_len]; char_idx: [batch_size, seq_len, word_len]
        embed = self.embedding(word_idx, char_idx)          # embed: [batch_size, seq_len, hidden_size]
