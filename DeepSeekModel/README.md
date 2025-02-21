# DeepSeek Training Implementation

This repository contains a PyTorch implementation of the DeepSeek model architecture. It enhances the SmolLM2 base implementation by incorporating DeepSeek's advanced architectural features, including Multi-Head Latent Attention (MHLA) and Mixture of Experts (MoE). The model is trained using the "HuggingFaceTB/smollm-corpus" dataset (cosmopedia-v2 configuration) with a streaming data loading approach.

## Model Architecture

### Key Components
- **Base Architecture:** Transformer-based decoder-only model
- **Model Size:** 774M parameters (1.48 GB)
- **Context Length:** 2048 tokens

### Advanced Features
- **Multi-Head Latent Attention (MHLA):** 8x compression ratio
- **Mixture of Experts (MoE):**
  - 8 expert networks
  - 1 shared expert
  - Top-2 expert routing
- **Rotary Position Embeddings (RoPE)**
- **Group Query Attention (GQA)**
- **RMSNorm** for layer normalization

### Configuration Details
- **Hidden Size:** 576
- **Intermediate Size:** 1536
- **Number of Attention Heads:** 9
- **Number of Key-Value Heads:** 3
- **Number of Layers:** 30
- **Vocabulary Size:** 49,152

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

### Training Configuration
The model is trained with the following specifications:
- **Training Steps:** 10,000 + 100 extended steps
- **Micro Batch Size:** 5120
- **Batch Accumulation Steps:** 2
- **Sequence Length:** 2048
- **Learning Rate:** 0.003 with linear warmup (2000 steps) and decay
- **Weight Decay:** 0.01
- **Gradient Clipping:** 1.0
- **Checkpointing Interval:** 500 steps
- **Text Generation Interval:** 500 steps
- **Mixed Precision:** bfloat16 (on supported devices)
- **Optimizer:** AdamW with parameters β₁ = 0.9, β₂ = 0.95, ε = 1e-8

### Training Configuration
During training, the model processes batches of data, computes the loss, and updates the model weights. The training script generates a detailed log file capturing key metrics such as loss values, learning rates, step times, and tokens processed per second. This log file is invaluable for monitoring training progress and diagnosing any issues.

Example log output:
```
Step 10001/10100 | Loss: 0.4346 | LR: 0.003000 | Total Step Time: 6734.82ms | Tokens/sec: 1556947.13 (accumulated over 5120 batches)
```
You can find the [Training log file:](https://github.com/Ezhirko/Creative-AI-apps/blob/main/DeepSeekModel/Traininglogs.txt) here !

### Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License.

### Acknowledgements
- **DeepSeek:** For the architectural inspiration behind this implementation.
- **SmolLM2:** For providing the robust base implementation.
- **HuggingFaceTB/smollm-corpus:** For the dataset used in training.
