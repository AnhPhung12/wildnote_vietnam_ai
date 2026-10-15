from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import os
import numpy as np
from core.vit_model import ai_extractor

app = FastAPI(title="Wildnote AI Online Space")

# Cho phép App React Native gọi vào API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp file vectors_database.json lên RAM khi khởi động
VECTOR_DB_PATH = "data/vectors_database.json"
database_vectors = []

if os.path.exists(VECTOR_DB_PATH):
    with open(VECTOR_DB_PATH, "r") as f:
        database_vectors = json.load(f)
    print(f" Successfully loaded {len(database_vectors)} vectors into Hugging Face!")
else:
    print("Could not find vectors_database.json in the data folder!")

@app.get("/")
def home():
    return {"status": "Wildnote AI Online Service is running perfectly!"}

@app.post("/identify")
async def identify_image(file: UploadFile = File(...)):
    if not database_vectors:
        raise HTTPException(status_code=500, detail="Vector database is not ready.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Trích xuất vector từ ảnh người dùng gửi lên bằng ViT
        query_vector = ai_extractor.extract_vector(image)
        q_vec = np.array(query_vector)
        
        # 2. Quét tìm kiếm Cosine Similarity với 4.627 vector
        best_match = None
        highest_similarity = -1.0
        
        for item in database_vectors:
            db_vec = np.array(item["vector"])
            similarity = np.dot(q_vec, db_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(db_vec))
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = item["species_id"]

        return {
            "status": "success",
            "species_id": best_match,
            "confidence": float(highest_similarity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))