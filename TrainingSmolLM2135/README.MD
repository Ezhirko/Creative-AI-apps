### SmolLM2 Training Implementation
This repository contains a PyTorch implementation of the SmolLM2-135M model training pipeline. The model is trained on the "HuggingFaceTB/smollm-corpus" dataset (cosmopedia-v2 configuration) using streaming data loading.

##### Model Architecture
SmolLM2 is a 135M parameter language model with the following specifications:

30 transformer layers
576 hidden dimension
9 attention heads
3 key-value heads
1536 intermediate size
49,152 vocabulary size
RMSNorm for layer normalization
SiLU activation function
Rotary positional embeddings

