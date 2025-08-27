# Face Verification with Blink Detection & Anti-Spoofing

This project provides a **face verification system** with the following features:

- **Frontend (Web App)** built with HTML, CSS, and JavaScript.
  - Webcam access with **MediaPipe Vision Tasks** for face detection, face landmarking, blink detection, and face covering classification (plain/covered).
  - Quality checks for blur, brightness, centering, and multiple faces.
  - Timer-based face capture with blink validation.
  - Uploads the final captured frame to the backend for verification.

- **Backend (FastAPI)** with Python:
  - Stores face embeddings in a **ChromaDB** vector database.
  - Uses **DeepFace (ArcFace model)** for embeddings.
  - Performs **anti-spoofing** detection.
  - Verifies faces with cosine similarity.
  - Provides REST API endpoints.


## 📂 Project Structure

```

.
├── backend/                # FastAPI backend
│   ├── main.py             # API routes
│   ├── face\_recognition.py # Embedding + verification
│   ├── face\_anti\_spoofing.py # Spoofing detection
│   ├── chromadb\_functions.py # Database operations
│
├── frontend/               # Frontend client
│   ├── index.html          # UI
│   ├── app.js              # Face detection + capture
│
├── web\_model/              # Model storage (if needed later)
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── LICENCE
└── .gitignore

````


## 🚀 Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/face-verification-app.git
cd face-verification-app
````

### 2. Backend Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
cd backend
uvicorn main:app --reload
```

The backend will run at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 3. Frontend Setup

Simply open `frontend/index.html` in a browser (Chrome recommended).

Or use a simple server:

```bash
cd frontend
python -m http.server 8080
```

Then visit: **[http://localhost:8080](http://localhost:8080)**


## 📌 API Endpoints

### **1. Upload Face to Database**

Registers a user’s face.

```http
POST /upload_face/
Form Data:
- name: string (username)
- image: file (face image)
```

### **2. Face Matching & Verification**

Verifies a face against the stored embedding.

```http
POST /face_matching/
Form Data:
- name: string (username)
- image: file (captured image)
```

**Response Example**

```json
{
  "status": true,
  "message": "Success",
  "num_face": 1,
  "file_size_bytes": 183243,
  "verified": true,
  "distance": 0.3234,
  "similarity_percentage": 67.66,
  "is_face_real": true
}
```


## ⚙️ Features

```
✅ Face Detection & Blink Detection (MediaPipe)
✅ Face Covering Classification (Plain / Covered)
✅ Blur & Low Light Detection
✅ Face Embedding & Verification (DeepFace ArcFace)
✅ Anti-Spoofing Detection
✅ Vector DB storage with **ChromaDB**
```

## 📦 Requirements

* Python 3.8+
* FastAPI
* Uvicorn
* DeepFace
* ChromaDB
* Mediapipe (frontend via CDN)

Install via:

```bash
pip install -r requirements.txt
```


## 📖 Future Improvements

* Multi-user enrollment & management
* Docker setup for backend + frontend
* Cloud deployment
* WebSocket streaming instead of HTTP polling


## 📝 License

This project is licensed under the **MIT License**.





