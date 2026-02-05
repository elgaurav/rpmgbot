import os
import shutil
from pathlib import Path
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import PictureItem, TableItem # <--- NEW IMPORT NEEDED

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
# Pointing to your TEST DATA folder first
DATA_DIR = BASE_DIR.parent / "data" / "piping" / "test_data"
STORAGE_DIR = BASE_DIR / "storage"
# Where images will be physically saved
IMAGE_OUTPUT_DIR = BASE_DIR / "static" / "images"

# Ensure output directories exist
if STORAGE_DIR.exists():
    try:
        shutil.rmtree(STORAGE_DIR) # Clean old index
    except Exception as e:
        print(f"⚠️ Warning: Could not delete storage dir (might be open): {e}")

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_ingestion():
    print(f"🚀 Starting Ingestion on TEST DATA: {DATA_DIR}")
    
    # 1. SETUP MODELS (GPU Enabled)
    print("⚙️  Initializing AI Models...")
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = Ollama(model="llama3.2", request_timeout=300.0) 

    # 2. CONFIGURE DOCLING
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True # <--- Essential for extraction
    pipeline_options.images_scale = 2.0 

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 3. PROCESS FILES
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_DIR}")
        return

    documents = []
    print(f"📸 Found {len(pdf_files)} PDF(s). Processing with Vision...")

    for pdf_path in pdf_files:
        print(f"   👉 Processing: {pdf_path.name}")
        
        # Convert PDF
        conv_result = converter.convert(pdf_path)
        doc_filename = pdf_path.stem
        counter = 0
        
        # --- FIXED LOOP LOGIC ---
        # We iterate through all items and check if they are Pictures
        for element, _level in conv_result.document.iterate_items():
            
            if isinstance(element, PictureItem):
                counter += 1
                image_name = f"{doc_filename}_fig_{counter}.png"
                image_path = IMAGE_OUTPUT_DIR / image_name
                
                # Get the image data
                img_data = element.get_image(conv_result.document)
                
                if img_data:
                    with open(image_path, "wb") as f:
                        img_data.save(f, "PNG")
                    print(f"      🖼️  Extracted: {image_name}")

        # Export Text to Markdown
        md_text = conv_result.document.export_to_markdown()
        
        # Create a LlamaIndex Document
        doc = Document(text=md_text, metadata={"file_name": pdf_path.name})
        documents.append(doc)

    # 4. CREATE VECTOR INDEX
    print("🧠 Building vector database...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 5. SAVE
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print(f"🎉 Database saved! Images are in: {IMAGE_OUTPUT_DIR}")

if __name__ == "__main__":
    run_ingestion()