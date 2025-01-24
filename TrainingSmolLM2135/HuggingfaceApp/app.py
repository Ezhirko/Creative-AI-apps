import gradio as gr
import torch
from transformers import AutoTokenizer
from smolLM2 import SmolLM2
from huggingface_hub import hf_hub_download

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")  # Adjust if necessary

# Load model configuration
model_config = {
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "silu",
    "hidden_size": 576,
    "initializer_range": 0.041666666666666664,
    "intermediate_size": 1536,
    "is_llama_config": True,
    "max_position_embeddings": 2048,
    "num_attention_heads": 9,
    "num_hidden_layers": 30,
    "num_key_value_heads": 3,
    "pad_token_id": None,
    "pretraining_tp": 1,
    "rms_norm_eps": 1.0e-05,
    "rope_interleaved": False,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 49152
}

# Initialize the model with the configuration
model = SmolLM2(model_config)  # Pass the configuration to the model

# Replace with your Hugging Face model ID (username/model-name)
model_id = "EzhirkoArulmozhi/SmolLM2-135"
model_path = hf_hub_download(repo_id=model_id, filename="smollm2_final.pt")
# Load the model weights with map_location to handle CPU-only environments
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the model weights
model.eval()  # Set the model to evaluation mode

def generate_text(prompt, length, num_sequences):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    generated_texts = []
    for _ in range(num_sequences):
        generated_sequence = model.generate(
            input_ids,
            max_length=length + len(input_ids[0]),  # Adjust for input length
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        
        # Decode the generated sequence
        generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    # Format the output
    formatted_output = "\n\n".join([f"Sequence {i + 1}:\n{text}" for i, text in enumerate(generated_texts)])
    return formatted_output

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# SmolLM2 Text Generator")
    prompt_input = gr.Textbox(label="Enter your text prompt", placeholder="Type your prompt here...")
    length_slider = gr.Slider(minimum=10, maximum=200, label="Predict Additional Text of Length", value=50)
    num_sequences_slider = gr.Slider(minimum=1, maximum=5, label="Number of Sequences to Generate", value=1, step=1)  # Step set to 1 for integer values
    generate_button = gr.Button("Generate Text")
    output_text = gr.Textbox(label="Generated Text", interactive=False)

    generate_button.click(
        fn=generate_text,
        inputs=[prompt_input, length_slider, num_sequences_slider],
        outputs=output_text
    )

# Launch the app
app.launch()