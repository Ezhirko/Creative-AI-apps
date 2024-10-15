# Resume Optimizer Chrome Extension 🚀

### A Smart Job Search Tool to Optimize Your Resume for Job Applications

---

## Overview

The **Resume Optimizer Chrome Extension** helps job seekers **enhance** their resumes by providing real-time insights on how well their resume matches a specific job description.

With this tool, users can:

- **Upload their resume** directly through the extension.
- Select any **job description** from popular job portals.
- Instantly get a **match percentage**, indicating how closely their resume fits the job requirements.
- Receive a **list of missing keywords** that are crucial to tailoring their resume for a higher chance of passing ATS (Applicant Tracking Systems) filters.

---

## Key Features

- 📊 **Job Match Percentage**: Get a detailed score showing how well your resume fits the job description.
- 🔑 **Missing Keywords**: Identify the essential keywords from the job description that are missing from your resume.
- ⚡ **Real-Time Results**: Analyze your resume against job descriptions in just seconds.
- 🧠 **AI-Powered**: The backend server leverages **Google's Gemini Pro AI model** to process your resume and job descriptions.

---

## How It Works

1. **Client-Server Architecture**: The Chrome extension communicates with a **Python-based Flask server**.
2. The user uploads their resume and selects the job description from the browser.
3. The resume text and job description are sent to the Flask server.
4. **Google's Gemini Pro model** analyzes the resume and returns the job match percentage and the missing keywords.
5. The result is displayed in a popup dialog within the Chrome extension for easy viewing.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/resume-optimizer-extension.git
2. **Navigate to the project directory**:
   ```bash
   cd resume-optimizer-extension
3. **Install necessary Python packages:**
   ```bash
   pip install -r requirements.txt
4. **Run the Flask server:**
   ```bash
   python app.py
5. **Load the Chrome extension:**
   -Open Chrome and go to chrome://extensions/.
   -Enable Developer mode.
   -Click Load unpacked and select the extension directory.

---

## Usage
1. Upload your resume through the extension.
2. Navigate to a job portal, select the job description, and click the extension button.
3. Instantly receive your resume match percentage and missing keywords!

[Check out the demo on YouTube](https://youtu.be/6ORtHFWQ6fM?si=ekDYrLU1hqM4fybM)
