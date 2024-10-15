from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

genai.configure(api_key="AIzaSyAHPgj12oMi9nH54YHFpN869-SvfkE415w")

def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

input_prompt = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Jd and
the missing keywords with high accuracy
resume:{resumeText}
description:{jobDescription}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords":["keyword1","keyword2","keyword3"]}}
"""

@app.route('/concatenate', methods=['POST'])
def concatenate():
    data = request.json
    jobDescription = data.get('JobDescription', '')
    resumeText = data.get('ResumeText', '')
    #concat_name = concatenate_names(first_name, second_name)
    formatted_prompt = input_prompt.format(resumeText=resumeText,jobDescription=jobDescription)
    response = get_gemini_response(formatted_prompt)
    return jsonify({'full_name': response})

def concatenate_names(first_name, second_name):
    return f"{first_name} {second_name}"

if __name__ == '__main__':
    app.run(debug=True)
