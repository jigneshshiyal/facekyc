from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os 
import shutil
from chromadb_functions import add_face_in_db, get_face_emb_from_db
from face_recognition import get_face_embedding, verify_faces
from face_anti_spoofing import check_face_anti_spoofing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_face/")
async def upload_face(name: str = Form(...), image: UploadFile = File(...)):
    suffix = os.path.splitext(image.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(image.file, tmp_file)
    try:
        
        file_size = os.path.getsize(tmp_file.name)

        add_facedb_status, num_face, message = add_face_in_db(tmp_path, name)

        return JSONResponse({
            "status": add_facedb_status,
            "message": message, 
            "num_face": num_face,
            "file_size_bytes": file_size,
            "temp_file": tmp_path,
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/face_matching/")
async def face_matching(name: str = Form(...), image: UploadFile = File(...)):
    suffix = os.path.splitext(image.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(image.file, tmp_file)
    try:
        verified = 0
        distance = 0
        similarity_percentage = 0
        message = ""
        is_face_real = False

        file_size = os.path.getsize(tmp_file.name)

        face_emb_status, image_embedding, num_face, warning = get_face_embedding(tmp_path)
        message += ",  " + warning

        if face_emb_status == False:
            pass
        else:
            anti_spoofing_status, is_face_real, spoofing_message = check_face_anti_spoofing(tmp_path)
            message += ",  " + spoofing_message

            if anti_spoofing_status:
                get_emb_db_status, save_emb, db_get_message = get_face_emb_from_db(name)
                message += ",  " + db_get_message

                if get_emb_db_status:
                    face_verify_result = verify_faces(image_embedding, save_emb.tolist()[0])
                    if face_verify_result["status"]==True:
                        verified = bool(face_verify_result["verified"])
                        distance = float(face_verify_result["distance"])
                        similarity_percentage = float(face_verify_result["similarity_percentage"])

        return JSONResponse({
            "status": True,
            "message": message, 
            "num_face": num_face,
            "file_size_bytes": file_size,
            "temp_file": tmp_path,
            "verified":verified, 
            "distance":distance,
            "similarity_percentage": similarity_percentage,
            "is_face_real": is_face_real
        })
    except Exception as e:
        return JSONResponse({
            "status": False,
            "message": message, 
            "num_face": num_face,
            "file_size_bytes": file_size,
            "temp_file": tmp_path,
            "verified":verified, 
            "distance":distance,
            "similarity_percentage": similarity_percentage,
            "is_face_real": is_face_real
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



if __name__=="__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)