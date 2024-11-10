#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Marius Kurz
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# 1. The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# 2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""This module implements the transformer model.

This module implements the transformer model as described in the original
paper "Attention is All You Need" by Vaswani et al. (2017) and using the
building blocks from the `torchformer` package. 

The model itself expects already tokenized input and output sequences and the
right-shifting of the target sequence should be performed before passing it to
the model. The model will then return the output probabilities per token.
"""

import torch
from torch import nn

from torchformer import TransformerEncoderBlock
from torchformer import TransformerDecoderBlock
from torchformer import PositionalEncoding

class Transformer(nn.Module):
    """The Transformer model.

    This class implements the transformer model as described in the original
    paper "Attention is All You Need" by Vaswani et al. (2017) as closely as
    possible. The model expects already tokenized input and output sequences
    and the right-shifting of the target sequence should be performed before
    passing it to the model. The model will then return the output
    probabilities per token.

    Args:
        num_blocks (int): The number of encoder and decoder blocks.
        vocab_size (int): The size of the vocabulary.
        d_model (int): The latent embedding dimension.
        num_heads (int): The number of attention heads.
        d_k (int): The dimensionality of queries and keys in attention.
        d_v (int): The dimensionality of values in attention.
        d_ff (int): Number of neurons in hidden layer of feed-forward networks.
        dropout (float): The dropout rate.
    """
    def __init__(
            self,
            num_blocks,
            vocab_size,
            context_length,
            d_model=512,
            num_heads=8,
            d_k=None,
            d_v=None,
            d_ff=2048,
            dropout=0.1,
        ):
        super().__init__()

        # Set default values
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads

        # Create the encoder and decoder blocks
        encoder_list = []
        decoder_list = []
        for _ in range(num_blocks):
            encoder_list.append(
                TransformerEncoderBlock(d_model, num_heads, d_k, d_v, d_ff, dropout)
            )
            decoder_list.append(
                TransformerDecoderBlock(d_model, num_heads, d_k, d_v, d_ff, dropout)
            )
        self.encoder = nn.ModuleList(encoder_list)
        self.decoder = nn.ModuleList(decoder_list)

        # Get Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, context_length)

        # Create dropout layer for encoder and decoder
        self.dropout_enc = nn.Dropout(dropout)
        self.dropout_dec = nn.Dropout(dropout)

        # Create the (shared) embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, y):
        """Perform a forward pass through the network.

        This implements all operations shown in Figure 1 of the original
        Transformer paper. This include in particular the steps:
            1. Embedding and positional encoding of the input sequence.
            2. Passing the input through the encoder.
            3. Embedding and positional encoding of the output sequence.
            4. Passing the output through the decoder.
            5. Applying the output projection and softmax to the decoder output.

        Args:
            x (torch.Tensor): The tokenized input sequence.
            y (torch.Tensor): The tokenized (right-shifted) target sequence.

        Returns:
            torch.Tensor: The output probabilities per token.
        """
        # 1. Run the input through the embedding and encoder
        x = self.embedding(x)
        x = self.dropout_enc(x)
        for block in self.encoder:
            x = block(x)

        # 2. Run the output through the embedding and decoder
        y = self.embedding(y)
        y = self.dropout_dec(y)
        for block in self.decoder:
            y = block(y, x)

        # 3. Apply the output projection and softmax
        y = self._output_projection(y)
        y = torch.softmax(y, dim=-1)

        return y

    def _output_projection(self, x):
        """Apply the output projection with shared embedding weights.

        Args:
            x (torch.Tensor): The decoder output.

        Returns:
            torch.Tensor: The output probabilities per token.
        """
        return torch.matmul(x, self.embedding.weight.T) + self.output_bias
