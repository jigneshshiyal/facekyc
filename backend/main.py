from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/model")
def get_model():
    model_path = os.path.join("./web_model", "face_covering_cls_model_int8.tflite")
    return FileResponse(model_path, media_type="application/octet-stream")

def main():
    uvicorn.run(app=app, port=8000)


main()