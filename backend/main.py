import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from engine import query_piping_data

app = FastAPI(title="RPMG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# MOUNT IMAGES so http://localhost:8000/static/images/x.png works
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_chat():
    index = FRONTEND_DIR / "index.html"
    return FileResponse(index) if index.exists() else HTTPException(404)

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    print(f"User asked: {req.question}")
    try:
        result = query_piping_data(req.question)
        
        # Convert filenames to Full URLs
        image_urls = []
        if "images" in result:
            for img_name in result["images"]:
                # Create the clickable link
                full_url = f"http://127.0.0.1:8000/static/images/{img_name}"
                image_urls.append(full_url)
        
        result["images"] = image_urls
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))