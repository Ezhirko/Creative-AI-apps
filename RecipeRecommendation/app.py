from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')
lang_model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{
            "mime_type":uploaded_file.type,
            "data": bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

st.set_page_config(page_title="Recipe Recommender")
st.header("Recipe Recommender")
question = st.text_input("customise your recipe... ",key="input")
uploaded_file = st.file_uploader("Upload your grocery Image...",type=['jpg','jpeg','png'])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image.", use_column_width=True)

submit = st.button("Recomend a recipe")
input_prompt="""
You are an expert in identifying vegetables and groceries from a image. 
we will upload a image of a stocked up refrigerator and you will have to identify the 
vegetables and groceries items form the image and list them under the section "Items":
You will have to provide with step by step process of how to cook an indian recipes 
that is possible only with items identified from the image as ingredients. 
Provide Name of the recipe under the section "Recipe Name: followed by step to make it."""

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt,image_data,question)
    st.subheader("The Response is:")
    st.write(response)