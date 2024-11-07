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


"""This module implements the basic building blocks of the transformer model.

This module implements in PyTorch the basic building blocks of the transformer
model as proposed by Vaswani et al. in the paper "Attention is All You Need".

Classes:
    FullyConnectedLayer: Two fully-connected layers with activation in between.
    DotProductAttention: Dot-product attention mechanism.
    MultiHeadAttention: Multi-head version of the dot-product attention.
    TransformerEncoderBlock: A single encoder block of the transformer model.
    TransformerDecoderBlock: A single decoder block of the transformer model.
    PositionalEncoding: Positional encoding of the input embeddings.
"""

import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """Fully-Connected Layer.

    This module implements a simple fully-connected layer consisting of two
    linear transformations with an activation function in between.

    Args:
        d_model (int): The input and output dimensionality.
        d_ff (int): The number of neurons in the hidden layer.
        activation (torch.nn.Module): The activation function to use.
    """
    def __init__(self, d_model, d_ff=None, activation=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Compute the forward pass through the layer.

        Args:
            x (torch.Tensor): The input to the layer.

        Returns:
            torch.Tensor: The output of the layer.
        """
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.fc2(x)


class DotProductAttention(nn.Module):
    """Dot-Product Attention.

    This module performs the dot-product attention mechanism. It projects the
    input to query, key and value vectors and computes the attention scores
    via matrix multiplication. Optionally, a mask can be applied to the
    attention scores before the softmax operation to avoid attending to
    later positions in the sequence.

    Args:
        d_model (int): The input dimensionality.
        d_k (int): The dimensionality of the query and key vectors.
        d_v (int): The dimensionality of the value vectors.
        mask (bool): Whether to apply a mask to the attention scores.
    """
    def __init__(self, d_model, d_k, d_v, mask=False):
        super().__init__()
        self.mask = mask
        self.d_k = d_k
        # create weight metrices for query, key and value
        self.w_q = nn.Parameter(torch.randn(d_model, d_k))
        self.w_k = nn.Parameter(torch.randn(d_model, d_k))
        self.w_v = nn.Parameter(torch.randn(d_model, d_v))

    def forward(self, x_q, x_k=None, x_v=None):
        """Compute the attention scores and weighted values.

        Args:
            x_q (torch.Tensor): Input used to compute queries.
            x_k (torch.Tensor): (Optional) Separate input used for keys.
            x_v (torch.Tensor): (Optional) Separate Input used for values.

        Returns:
            torch.Tensor: The weighted values.
            torch.Tensor: The attention scores.
        """
        # If key and value are not provided, assume self-attention
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q
        # First project input linearly to query, key and value
        Q, K, V = self._projection(x_q, x_k, x_v)
        # Compute attention scores for each query-key pair
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5
        # Masking: Set upper triangular matrix to -inf before softmax
        # Upper triangle since query i should only attend to prior keys j<=i
        if self.mask:
            mask = torch.tril(torch.ones_like(scores))
            scores = scores.masked_fill(mask == 0, -float('inf'))
        # Probabilities via Softmax (row-wise, i.e. per query)
        attention = torch.softmax(scores, dim=-2)
        # Weight values with attention scores
        return torch.matmul(attention, V), attention

    def _projection(self, x_q, x_k, x_v):
        """Perform projection with query, key and value weights.

        Args:
            x_q (torch.Tensor): Input used to compute queries.
            x_k (torch.Tensor): Separate input used for keys.
            x_v (torch.Tensor): Separate Input used for values.

        Returns:
            torch.Tensor: The projected queries.
            torch.Tensor: The projected keys.
            torch.Tensor: The projected values.
        """
        Q = torch.matmul(x_q, self.w_q)
        K = torch.matmul(x_k, self.w_k)
        V = torch.matmul(x_v, self.w_v)
        return Q, K, V


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    This module implements the multi-head attention mechanism. It consists
    of multiple attention heads, each of which computes attention scores
    independently. The outputs of the heads are concatenated and linearly
    transformed to the desired output dimensionality.

    Args:
        d_model (int): The embedding dimensionality.
        num_heads (int): The number of attention heads.
        d_k (int): The dimensionality of the query and key vectors.
        d_v (int): The dimensionality of the value vectors.
        mask (bool): Whether to apply masking to the attention scores.
    """
    def __init__(self, d_model, num_heads=1, d_k=None, d_v=None, mask=False):
        super().__init__()

        # Set default values
        self.num_heads = num_heads
        if d_k is None:
            d_k = d_model // self.num_heads
        if d_v is None:
            d_v = d_model // self.num_heads

        # Create multiple attention heads
        for i in range(num_heads):
            self.add_module(f'att_{i}', DotProductAttention(d_model, d_k, d_v, mask))

        # Add final projection layer. (Kinda lengthy) Explanation:
        #   This matrix maps for each head the low-dimension values to the
        #   embedding size, i.e. maps each V from d_v to d_model. Then all
        #   updates across all heads are simply summed up. Funnily enough these
        #   two independent operations can be combined to a single matrix
        #   vector product with the concatenated outputs of the heads.
        self.w_out = nn.Parameter(torch.randn(self.num_heads * d_v, d_model))

    def forward(self, x_q, x_k=None, x_v=None):
        """Compute the multi-head attention.

        Args:
            x_q (torch.Tensor): Input used to compute queries.
            x_k (torch.Tensor): (Optional) Separate input used for keys.
            x_v (torch.Tensor): (Optional) Separate Input used for values.

        Returns:
            torch.Tensor: The output of the multi-head attention.
        """
        head_outputs = []
        for i in range(self.num_heads):
            head_output, _ = getattr(self, f'att_{i}')(x_q, x_k, x_v)
            head_outputs.append(head_output)
        return torch.matmul(torch.cat(head_outputs, dim=-1), self.w_out)


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block.

    This module implements a single encoder block of the transformer model.
    It consists of two sub-layers:
        1. a multi-head self-attention layer
        2. a feed-forward network.
    Each of these sub-layers is followed by a layer normalization and a
    residual connection.

    Args:
        d_model (int): The embedding dimensionality.
        num_heads (int): The number of attention heads.
        d_k (int): The dimensionality of the query and key vectors.
        d_v (int): The dimensionality of the value vectors.
        d_ff (int): The number of neurons in the hidden layer of the
            feed-forward network.
        dropout (float): The dropout rate for the residual connections
    """
    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 d_k=None,
                 d_v=None,
                 d_ff=2048,
                 dropout=0.1
                ):
        super().__init__()

        # Set default values
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads

        # Create sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FullyConnectedLayer(d_model, d_ff, activation=nn.ReLU())
        self.norm2 = nn.LayerNorm(d_model)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        """Forward pass through the encoder block.

        Args:
            x (torch.Tensor): The input to the encoder block.

        Returns:
            torch.Tensor: The output of the encoder block.
        """
        # First block: Self-Attention
        x1 = self.attention(x)
        if self.dropout1 is not None:
            x1 = self.dropout1(x1)
        x  = self.norm1(x + x1)

        # Second block: Feed-Forward Network
        x1 = self.ff(x)
        if self.dropout2 is not None:
            x1 = self.dropout2(x1)
        x = self.norm2(x + x1)

        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block.

    This module implements a single decoder block of the transformer model.
    It consists of three sub-layers:
        1. a masked multi-head self-attention layer
        2. an encoder/decoder-attention layer
        3. a feed-forward network.
    Each of these sub-layers is followed by a layer normalization and a
    residual connection.

    Args:
        d_model (int): The embedding dimensionality.
        num_heads (int): The number of attention heads.
        d_k (int): The dimensionality of the query and key vectors.
        d_v (int): The dimensionality of the value vectors.
        d_ff (int): The number of neurons in the hidden layer of the
            feed-forward network.
        dropout (float): The dropout rate for the residual connections.
    """
    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 d_k=None,
                 d_v=None,
                 d_ff=2048,
                 dropout=0.1
                ):
        super().__init__()

        # Set default values
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads

        # Create sub-layers
        self.masked_attention = MultiHeadAttention(d_model, num_heads, d_k, d_v, mask=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FullyConnectedLayer(d_model, d_ff, activation=nn.ReLU())
        self.norm3 = nn.LayerNorm(d_model)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None
            self.dropout3 = None

    def forward(self, x, x_enc):
        """Forward pass through the decoder block.

        Args:
            x (torch.Tensor): The input from previous decoder block.
            x_enc (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The output of the decoder block.
        """
        # First block: Masked Self-Attention
        x1 = self.masked_attention(x)
        if self.dropout1 is not None:
            x1 = self.dropout1(x1)
        x  = self.norm1(x + x1)

        # Second block: Encoder/Decoder-Attention
        x1 = self.attention(x, x_k=x_enc, x_v=x_enc)
        if self.dropout2 is not None:
            x1 = self.dropout2(x1)
        x = self.norm2(x + x1)

        # Third block: Feed-Forward Network
        x1 = self.ff(x)
        if self.dropout3 is not None:
            x1 = self.dropout3(x1)
        x = self.norm3(x + x1)

        return x


class PositionalEncoding(nn.Module):
    """Positional Encoding.

    This module implements the positional encoding as proposed by Vaswani et al.
    in the paper "Attention is All You Need". It adds sinusoidal positional
    encodings to the input embeddings to provide information about the position
    of the tokens in the sequence.

    Args:
        d_model (int): The embedding dimensionality.
        context_size (int): The maximum length of the input sequence.
    """
    def __init__(self, d_model, context_length=512):
        super().__init__()
        # Compute positional encodings
        div_term = torch.exp(torch.arange(0, d_model, 2) / d_model * -(torch.log(torch.tensor(10000.0))))
        pos = torch.arange(context_length).unsqueeze(1)
        pe = torch.zeros(context_length, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Register constant positional encodings as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): The input embeddings.

        Returns:
            torch.Tensor: The input embeddings with added positional encoding.
        """
        return x + self.pe[:x.size(0), :]
