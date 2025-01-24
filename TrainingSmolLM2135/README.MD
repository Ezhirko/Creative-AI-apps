### SmolLM2 Training Implementation
This repository contains a PyTorch implementation of the SmolLM2-135M model training pipeline. The model is trained on the "HuggingFaceTB/smollm-corpus" dataset (cosmopedia-v2 configuration) using streaming data loading.

##### Model Architecture
SmolLM2 is a 135M parameter language model with the following specifications:
- 30 transformer layers
- 576 hidden dimension
- 9 attention heads
- 3 key-value heads
- 1536 intermediate size
- 49,152 vocabulary size
- RMSNorm for layer normalization
- SiLU activation function
- Rotary positional embeddings

##### Training Configuration
The model is trained with the following specifications:
- Training steps: 5000 + 50 extended steps
- Batch size: 8
- Sequence length: 2048
- Learning rate: 0.003 with linear warmup and decay
- Weight decay: 0.01
- Gradient clipping: 1.0
- Checkpointing interval: 1000 steps
- Validation/Generation interval: 500 steps
- bfloat16 precision (on supported devices)
- AdamW optimizer with β1=0.9, β2=0.95

##### Features
- Multi-device support (CUDA, MPS, CPU)
- Streaming dataset loading for memory efficiency
- Automatic checkpoint resumption
- Regular model checkpointing
- Training progress metrics (loss, learning rate, step time, tokens/sec) displayed using print statements
- Generation samples during training
- FP32 gradient accumulation support
- Learning rate scheduling with warmup and decay
- Weight tying between input and output embeddings

##### Training Process
During training, the model processes batches of data, computes the loss, and updates the model weights. The training script generates a log file that captures key metrics such as loss values, learning rates, and tokens processed per second. This log file is useful for monitoring the training progress and diagnosing any issues that may arise.

##### Training Log File
The training log file is generated during the training process and contains detailed information about each training step. It includes metrics such as:

- Step number
- Loss value
- Learning rate
- Total step time
- Tokens processed per second

You can find the [training log](https://github.com/your-username/your-repository/blob/main/path/to/file) file in the project directory after running the training script.

