from scipy.spatial.distance import cosine
from deepface import DeepFace
import logging
from deepface import DeepFace

logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def get_face_embedding(image_path, model_name="ArcFace", detector_backend="opencv"):
    try:
        # Use DeepFace to extract embeddings
        embedding_objs = DeepFace.represent(img_path=image_path, 
                                            model_name=model_name, 
                                            detector_backend=detector_backend, 
                                            enforce_detection=True)

        if not embedding_objs:
            return False, None, 0, "No face detected"

        if len(embedding_objs) > 1:
            return False, None, len(embedding_objs), "Multiple faces detected"

        embedding = embedding_objs[0]["embedding"]
        return True, embedding, 1, "Success"

    except Exception as e:
        logging.error(f"Error in get_face_embedding: {e}")
        return False, None, 0, f"Error: {e}"


def verify_faces(emb1, emb2, threshold=0.6):
    try:
        distance = cosine(emb1, emb2)
        similarity_percentage = (1 - distance) * 100
        verified = distance < threshold

        return {
            "status": True,
            "verified": verified,
            "distance": float(distance),
            "similarity_percentage": round(similarity_percentage, 2),
            "message":"Success"
        }
    except Exception as e:
        return {
            "status": False,
            "verified": 0,
            "distance": 0,
            "similarity_percentage": 0,
            "message": f"Error: {e}"
        }
