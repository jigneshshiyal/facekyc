import chromadb
from face_recognition import get_face_embedding

client = chromadb.PersistentClient("./facedb")

collection = client.get_or_create_collection(name="face")

def add_face_in_db(image_path, unique_id):
    try:
        face_emb_status, image_embedding, num_face, warning = get_face_embedding(image_path)
        if face_emb_status == False:
            return False, num_face, warning
        
        collection.add(
            embeddings=image_embedding,
            ids=unique_id
        )

        return True, num_face, "Success"
    except Exception as e:
        return False, 0, f"Error in add_face_in_db function: {e}"
    
def get_face_emb_from_db(unique_id):
    try:
        face_emb = collection.get(ids=[unique_id], include=["embeddings"])["embeddings"]
        return True, face_emb, "Success"
    except Exception as e:
        return False, None, f"Error: {e}"
    