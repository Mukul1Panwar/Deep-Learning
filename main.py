from fastapi import FastAPI,UploadFile,File
import os
import shutil
import uuid
from app.inference import upload_image,rec,base_model,index

app = FastAPI(title="Fashion Image Recommendation API")

# create the empty dir. where the file stores
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True) 

@app.get('/')
def home():
    return{"message":"Home page of Fashion Similarity Model FastApi"}

@app.get('/about')
def about():
    return{"message":"About the model -- Pick the top 5 similar item"}

@app.get('/contact')
def contact():
    return{"message":"Contact Us :- 9836XXXXXX"}

@app.post("/predict")
def predict(file:UploadFile = File(...)): # upload an image in fastapi , it accept the data in same format
    ext = file.filename.split(".")[-1]
    temp_name = f"{uuid.uuid4()}.{ext}"   # it gives the unique name 
    temp_path = os.path.join(UPLOAD_DIR,temp_name)

    with open(temp_path,'wb') as f:   # converts the file into binary
        shutil.copyfileobj(file.file,f)
    
    input_emb = upload_image(temp_path,base_model)
    results = rec(input_emb,index)

    os.remove(temp_path)  # remove the image from dir.

    return{
        "recommendations":results
    }
