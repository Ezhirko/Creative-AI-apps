import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download
from transformers import GPT, GPTConfig  # Import your model class

# Load the model from Hugging Face Hub
def load_model_from_huggingface(device):
    # Replace with your Hugging Face model ID (username/model-name)
    model_id = "EzhirkoArulmozhi/DecoderTransformerModel"
    checkpoint_path = hf_hub_download(repo_id=model_id, filename="gpt_checkpoint.pth")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    return model

def generate_text(model, device, prompt, max_length=100, num_samples=1):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            if tokens.size(1) >= 1024:  # GPT context length
                break
                
            logits = model(tokens)[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Remove special token check entirely
            # Just generate for the specified length or until context limit
    
    generated_texts = []
    for i in range(num_samples):
        text = enc.decode(tokens[i].tolist())
        generated_texts.append(text)
    
    return '\n\n---\n\n'.join(generated_texts)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model_from_huggingface(device)
# Force model to stay in eval mode
model.train(False)

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        model,
        device,
        gr.Textbox(label="Prompt", value="We are accounted poor citizens, the"),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Max Length"),
        gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Samples"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Shakespeare-style Text Generator",
    description="Enter a prompt to generate Shakespeare-style text continuation",
    examples=[
        ["O Romeo, Romeo, wherefore art thou", 100, 1],
        ["To be, or not to be, that is", 60, 2],
        ["Friends, Romans, countrymen, lend me", 50, 3],
        ["All the world's a stage, and all the", 100, 1],
        ["Now is the winter of our discontent", 100, 1],
        ["If music be the food of love,", 100, 1],
    ]
)

if __name__ == "__main__":
    iface.launch() 