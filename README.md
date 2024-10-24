# Cricket Personality Predictor - Decoder Only Transformer

A PyTorch implementation of a Decoder-Only Transformer model trained to predict cricket personalities based on questions.

## Project Overview

This project implements a Decoder-Only Transformer architecture to understand and predict cricket player associations based on specific questions. The model learns to map questions to their corresponding cricket personalities.

## Training Data Examples

The model is trained on question-answer pairs like:
- Q: "Who is the King of Cricket?" → A: "ViratKohli"
- Q: "Who is the BigShow of Cricket?" → A: "MaxWell"
- Q: "Who is the best test Captain in India?" → A: "ViratKohli"
- Q: "Who is the Universe Boss?" → A: "Chris Gayle"
- Q: "Who is the Alien in the Cricket Field?" → A: "ABdeVilliers"
- Q: "Who is the 360 degree player in the Cricket ?" → A: "ABdeVilliers"


## Technical Details

### Architecture
- Decoder-Only Transformer
- Built using PyTorch framework
- Self-attention mechanism
- Positional encoding
- Multi-head attention layers

### Model Components
- Token Embedding
- Positional Encoding
- Decoder Blocks
- Feed Forward Networks
- Output Linear Layer

```bash
python train.py
