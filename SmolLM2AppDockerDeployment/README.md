# 🚀 SmolLM2 LLM Deployment in Docker 🐳  

This repository demonstrates how to **deploy a 135M parameter SmolLM2 Large Language Model (LLM) in a Docker container** and set up another container to interact with it. Using **FastAPI**, we serve the model in one container while another container sends API requests, fetches the results, and displays them.

---

## 📌 Features  

✅ **Containerized SmolLM2 LLM** using Docker  
✅ **FastAPI-based API** for text generation  
✅ **Multi-container communication** using Docker Compose  
✅ **Real-time inference across containers**  
✅ **Step-by-step deployment guide**  

---

## 📂 Project Structure  
```bash
├── 📂 SmolLM2Server # Container 1: Hosting the LLM API
│ ├── Dockerfile
│ ├── main.py # FastAPI server to serve LLM
│ ├── requirements.txt
│ ├── model/ # Folder for storing the SmolLM2 model
│ ├── config.yaml
│ └── ...
├── 📂 TextGenerator # Container 2: Sending API requests
│ ├── Dockerfile
│ ├── main.py # FastAPI server to fetch results
│ ├── templates/ # HTML files for frontend
│ ├── requirements.txt
│ └── ...
├── docker-compose.yml # Defines services and networking
└── README.md # Project documentation
```

---

## 🚀 Setup & Installation  

### 🔹 Prerequisites  
Ensure you have the following installed on your system:  
- 🐳 **Docker**  
- 🐍 **Python 3.8+**  
- 📦 **pip & virtualenv**  

---

### 🔹 Step 1: Clone the Repository  

```bash
git clone https://github.com/Ezhirko/Creative-AI-apps.git
cd Creative-AI-apps
```

### 🔹 Step 2: Build & Run Docker Containers
```bash
 docker-compose up --build
```

This will:<br>
✅ **Build** the SmolLM2 model API container (app1) <br>
✅ **Build** the Text Generator container (app2) <br>
✅ **Start** both containers and link them using a Docker network <br>

### 🔹 Step 3: Access the Application
Once both containers are up and running:

- SmolLM2 API (Container 1) → http://localhost:8001
- Text Generator (Container 2) → http://localhost:8000
  
📢 Open http://localhost:8000 in your browser and interact with the LLM!

---
### 📌 API Endpoints
🔹 SmolLM2 API (Container 1)
| Method | Endpoint | Description                |
|--------|----------|----------------------------|
| POST   | /generate| Generates text from prompt |

🔹 Text Generator API (Container 2)
| Method | Endpoint | Description                      |
|--------|----------|----------------------------------|
| GET    | /        | UI to enter prompt & view output |
| POST   | /generate| Sends prompt to SmolLM2 API      |

### 📺 Video Demonstration
🎥 Watch the full deployment and setup guide here:
🔗 [YouTube Video Link](https://youtu.be/56EwkD_KcUA)



