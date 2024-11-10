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
from transformers import AutoTokenizer
from datasets import load_dataset

from torchformer import Transformer


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
MAX_LENGTH = 64       # Maximum sequence length
SEED = 42             # Random seed (only for shuffling)
NUM_TRAIN = int(1e5)  # Number of training samples
NUM_EPOCHS = 10       # Number of epochs
NUM_BLOCKS = 2        # Number of transformer blocks
BATCH_SIZE = 16       # Batch size
LOG_INTERVAL = 1      # Log interval 

def load_wm14(tokenize=True, num_train=None):
    # Load WMT2014 English-German dataset
    dataset = load_dataset("wmt14", "de-en").with_format("torch")
    
    # Access train, validation, and test splits
    if num_train is not None:
        train_data = dataset['train'].shuffle(seed=SEED).select(range(num_train))
    else:
        train_data = dataset['train'].shuffle(seed=SEED)
    val_data = dataset['validation']
    test_data = dataset['test']

    if tokenize:
        # Apply tokenizer
        def tokenize_wmt14(data):
            inputs = tokenizer(
                    data['translation']['de'],
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH
            )
            labels = tokenizer(
                    data['translation']['en'],
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH
            )
            return {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'labels': labels['input_ids']
            }

        train_data = train_data.map(tokenize_wmt14)
        val_data = val_data.map(tokenize_wmt14)
        test_data = test_data.map(tokenize_wmt14)
    return train_data, val_data, test_data
    

def train(n_epochs=10, batch_size=16):
    train_data, val_data, test_data = load_wm14(tokenize=True,
                                                num_train=NUM_TRAIN)


    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False
    )

    # Initialize the model
    model = Transformer(
            num_blocks=NUM_BLOCKS,
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            context_length=MAX_LENGTH
    )

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Initialize the loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) 

    for epoch_index in range(n_epochs):
        running_loss = 0.
        last_loss = 0.
        # Loop over pairs of input and target sequences
        for i, data in enumerate(training_loader):

            inputs = data['input_ids']
            labels = data['labels']

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs, labels)

            # Compute the loss and its gradients
            loss = loss_fn(
                    outputs.view(-1, tokenizer.vocab_size),
                    labels.view(-1)
            )
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                last_loss = running_loss / LOG_INTERVAL # loss per batch
                print(f'  batch {(i + 1):05d} loss: {last_loss:.5e}')
                running_loss = 0.

        # TODO: Add validation loss
        ## Every data instance is an input + label pair
        #print(tokenizer.decode(data['labels'][0], skip_special_tokens=True))
        #print(tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))

if __name__ == "__main__":
    train(n_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
