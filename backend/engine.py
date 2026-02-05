import re
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 1. SETUP: Use the Smart 3B Model (Matches your Ingestion)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2:1b", request_timeout=360.0)

_BACKEND_DIR = Path(__file__).resolve().parent
PERSIST_DIR = _BACKEND_DIR / "storage"

# This is where we look for images to verify they exist
STATIC_IMAGES_DIR = _BACKEND_DIR / "static" / "images"

def query_piping_data(question: str) -> dict:
    if not PERSIST_DIR.exists():
        return {"answer": "Error: Database missing. Run ingest_pro.py!", "sources": []}

    storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(question)
    
    sources = []
    found_images = set()
    
    # Get list of all actual images we saved during ingestion
    if STATIC_IMAGES_DIR.exists():
        existing_images = set(f.name for f in STATIC_IMAGES_DIR.glob("*.png"))
    else:
        existing_images = set()

    for node in response.source_nodes:
        file_name = node.metadata.get("file_name", "Unknown")
        page_label = node.metadata.get("page_label", "N/A")
        sources.append({"file": file_name, "page": page_label})
        
        # LOGIC: Return ALL images belonging to the retrieved source document.
        # This ensures if the user asks about a topic in "API 582", they get the relevant diagrams.
        doc_stem = Path(file_name).stem
        for img in existing_images:
            if img.startswith(doc_stem):
                found_images.add(img)

    return {
        "answer": str(response),
        "sources": sources,
        "images": list(found_images) # Sends filenames to main.py
    }