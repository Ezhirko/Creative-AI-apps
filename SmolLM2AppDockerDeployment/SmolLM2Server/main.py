from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from smolLM2 import SmolLM2
from huggingface_hub import hf_hub_download
from fastapi import Request

# Initialize FastAPI app
app = FastAPI(title="SmolLM2 Model Server")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/app/cosmo2-tokenizer")

# Model configuration
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

# Load the model
model = SmolLM2(model_config)
model_id = "EzhirkoArulmozhi/SmolLM2-135"
#model_path = hf_hub_download(repo_id=model_id, filename="smollm2_final.pt")
model_path = "/app/checkpoint/smollm2_final.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Request/response models
class GenerateRequest(BaseModel):
    prompt: str
    length: int = 50
    num_sequences: int = 1

class GenerateResponse(BaseModel):
    generated_text: str

@app.get("/")
async def ui(request: Request):
    return "Server is running"

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    try:
        input_ids = tokenizer(request.prompt, return_tensors="pt")["input_ids"]
        generated_texts = []

        for _ in range(request.num_sequences):
            generated_sequence = model.generate(
                input_ids,
                max_length=request.length + len(input_ids[0]),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            )
            generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        return GenerateResponse(generated_text="\n\n".join(generated_texts))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
