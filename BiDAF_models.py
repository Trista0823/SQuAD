"""Top-level BiDAF model class.

Author:
    Chris Chute (chute@stanford.edu), Mengyao Bao(mb7570@nyu.edu)
"""

import BiDAF_layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Modified Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices(lookup) to get word vectors and embed character indices(fine tune) with
                           Conv1d, then go throw a Highway and dropout.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.tensor): Pre-trained word vectors.
        char_vectors (torch.tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()

        # Char Embedding
        # self.emb = BiDAF_layers.CharEmbeddings(word_vectors=word_vectors,
        #                                        char_vectors=char_vectors,
        #                                        hidden_size=hidden_size,
        #                                        dropout_rate=drop_prob)

        self.emb = BiDAF_layers.WordEmbedding(word_vectors=word_vectors,
                                              hidden_size=hidden_size,
                                              drop_prob=drop_prob)

        self.enc = BiDAF_layers.RNNEncoder(input_size=hidden_size,
                                           hidden_size=hidden_size,
                                           num_layers=1,
                                           drop_prob=drop_prob)

        self.att = BiDAF_layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                               drop_prob=drop_prob)

        self.mod = BiDAF_layers.RNNEncoder(input_size=8 * hidden_size,
                                           hidden_size=hidden_size,
                                           num_layers=2,
                                           drop_prob=drop_prob)

        self.out = BiDAF_layers.BiDAFOutput(hidden_size=hidden_size,
                                            drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # charEmbedding
        # c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        # q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # wordEmbedding
        c_emb = self.emb(cw_idxs)
        q_emb = self.emb(qw_idxs)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
