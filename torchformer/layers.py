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


import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, d_model, d_ff=None, activation=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.fc2(x)


class DotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        # create weight metrices for query, key and value
        self.w_query = nn.Parameter(torch.randn(d_model, d_k))
        self.w_key = nn.Parameter(torch.randn(d_model, d_k))
        self.w_value = nn.Parameter(torch.randn(d_model, d_v))

    def forward(self, value_in):
        # First project input values to query, key and value
        value, query, key = self._projection(value_in)
        # Calculate attention scores
        print(value.shape, query.shape, key.shape)
        scores = torch.tensordot(query, key, dims=([-1], [-1]))
        scores *= 1. / key.size(-1) ** 0.5
        # Probabilities via Softmax
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

    def _projection(self, value_in):
        """Perform projection with query, key and value weights"""
        query = torch.matmul(value_in, self.w_query)
        key = torch.matmul(value_in, self.w_key)
        value = torch.matmul(value_in, self.w_value)
        return value, query, key


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, d_k=None, d_v=None):
        super().__init__()
        # Set default values
        self.num_heads = num_heads
        if d_k is None:
            d_k = d_model // self.num_heads
        if d_v is None:
            d_v = d_model // self.num_heads
        # Create multiple attention heads
        for i in range(num_heads):
            self.add_module(f'att_{i}', DotProductAttention(d_model, d_k, d_v))
        # Add final projection layer
        self.w_proj = nn.Parameter(torch.randn(self.num_heads * d_v, d_model))

    def forward(self, x):
        head_outputs = []
        for i in range(self.num_heads):
            head_output, _ = getattr(self, f'att_{i}')(x)
            head_outputs.append(head_output)
        return torch.matmul(torch.cat(head_outputs, dim=-1), self.w_proj)


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 d_k=None,
                 d_v=None,
                 d_ff=None,
                 dropout=0.1
                ):
        super().__init__()
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads
        if d_ff is None:
            d_ff = 4 * d_model
        self.attention = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FullyConnectedLayer(d_model, d_ff, activation=nn.ReLU())
        self.norm2 = nn.LayerNorm(d_model)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x1 = self.attention(x)
        if self.dropout is not None:
            x1 = self.dropout(x1)
        x  = self.norm1(x + x1)
        x1 = self.ff(x)
        if self.dropout is not None:
            x1 = self.dropout(x1)
        x = self.norm2(x + x1)
        return x
