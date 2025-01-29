from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import requests

# Initialize FastAPI app
app = FastAPI(title="SmolLM2 UI Server")

# Mount static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API endpoint to render the UI
@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint to handle text generation requests
@app.post("/generate")
async def generate_text(request: Request):
    form_data = await request.form()
    prompt = form_data.get("prompt")
    length = int(form_data.get("length", 50))
    num_sequences = int(form_data.get("num_sequences", 1))

    try:
        response = requests.post(
            "http://app1:8001/generate",  # URL of App 1
            json={"prompt": prompt, "length": length, "num_sequences": num_sequences},
            timeout=10  # Optional timeout for better error handling
        )

        if response.status_code == 200:
            generated_text = response.json().get("generated_text", "")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "prompt": prompt, "generated_text": generated_text},
            )
        else:
            return templates.TemplateResponse(
                "index.html", {"request": request, "error": response.text}
            )

    except requests.exceptions.RequestException as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": str(e)}
        )
