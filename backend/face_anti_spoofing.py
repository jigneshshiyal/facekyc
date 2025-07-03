from deepface import DeepFace

def check_face_anti_spoofing(image_path):
    try:
        face_objs = DeepFace.extract_faces(img_path=image_path, anti_spoofing = True)
        return True, face_objs[0]["is_real"], "Success"
    except Exception as e:
        return False, False, f"Error: {e}"