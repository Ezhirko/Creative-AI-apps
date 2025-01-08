# Tamil Language BPE Tokenizer Project

## What is Byte Pair Encoding (BPE)?

Byte Pair Encoding (BPE) is a subword tokenization algorithm that iteratively merges the most frequent pairs of bytes (or characters) in a dataset to form new subword tokens. The result is a compact vocabulary that balances efficiency and expressiveness. BPE is widely used in training transformers and modern NLP models for the following reasons:

1. **Reduction in Vocabulary Size**: By breaking text into subword units, BPE achieves a smaller vocabulary, reducing the memory footprint of the model.
2. **Handling Out-of-Vocabulary (OOV) Words**: BPE decomposes unknown words into known subword units, improving the model's ability to generalize to unseen data.
3. **Compression**: BPE helps compress textual data by using a smaller set of subwords while retaining the ability to reconstruct the original text.
4. **Improved Performance**: By finding an optimal balance between character-level and word-level representations, BPE enables better performance in downstream tasks.

BPE is often the first step in preparing textual data for transformer models like BERT, GPT, and others, as it provides a tokenized input suitable for embedding layers.

---

## Project Overview: Tamil Language BPE Tokenizer

### Dataset
The dataset used for this project is the **Tamil Language Corpus for NLP** from [Kaggle](https://www.kaggle.com/datasets/praveengovi/tamil-language-corpus-for-nlp). The dataset contains **1,048,108 tokens**.

### Tokenizer Implementation
- **Language**: Tamil
- **Algorithm**: Byte Pair Encoding (BPE)
- **Vocabulary Size**: **5000 tokens**
- **Compression Ratio**: **3.2 or above**
- **Achieved Compression Ratio**: **11.48%**

### Steps Performed
1. **Dataset Preprocessing**: Cleaned and prepared the dataset for tokenization.
2. **BPE Training**:
   - Trained a BPE tokenizer to encode Tamil text into subwords.
   - Limited the vocabulary to 5000 tokens for optimal size and performance.
3. **Evaluation**:
   - Calculated the compression ratio as the ratio of original tokens to the total number of tokens post-encoding.
   - Achieved a compression ratio of **11.48%**, which is significantly higher than the required **3.2%**.
4. **Deployment**:
   - Deployed the tokenizer on Hugging Face Spaces using **Gradio** as the user interface.

### Application Features
The deployed application provides:
- **Encoding**: Converts Tamil text into BPE tokens.
- **Decoding**: Reconstructs the original Tamil text from BPE tokens.
- **Visualization**: Demonstrates the tokenization process step-by-step for educational purposes.

### Project Deliverables
1. **GitHub Repository**: Contains the training notebook, BPE implementation, and README file with token count and compression ratio.
   - [GitHub Repository Link](https://github.com/your-repo-link)
2. **Hugging Face Spaces Deployment**:
   - Application deployed with examples for encoding and decoding Tamil text.
   - [Hugging Face Spaces Link](https://huggingface.co/spaces/EzhirkoArulmozhi/BPETokenizer-Tamil-Language)

### How to Use the Application
1. Enter Tamil text in the input field.
2. Click "Encode" to convert the text into BPE tokens.
3. View the encoded tokens and their mapping.
4. Use the "Decode" button to reconstruct the original text from the tokens.

---

This project demonstrates the utility of BPE in creating a compact and efficient tokenizer for the Tamil language, making it an essential tool for modern NLP workflows.
