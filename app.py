from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import random
import json
import openai
import requests
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

# Load API key from JSON file
with open('config.json') as f:
    config = json.load(f)
    api_key = config['openai_api_key']

# Set the API key for OpenAI
openai.api_key = api_key

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

@app.route('/generate_image', methods=['POST', 'OPTIONS'])
def generate_image():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        data = request.json
        animal = data.get('animal')
        prompt = data.get('prompt')

        if not prompt:
            # If prompt is empty, return a random image URL for the selected animal
            if animal == "Cat":
                dalle_prompt = "a beautiful looking cat."
            elif animal == "Dog":
                dalle_prompt = "a beautiful looking dog."
            elif animal == "Elephant":
                dalle_prompt = "a beautiful looking elephant."
        else:
            dalle_prompt = prompt
                    # Generate image using DALL·E
        dalle_response = openai.Image.create(
            prompt=dalle_prompt,
            size="512x512"
        )
        image_url = dalle_response['data'][0]['url']
        return _corsify_actual_response(jsonify({"image_url": image_url}))

def _build_cors_preflight_response():
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run(debug=True)
