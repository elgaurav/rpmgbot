"""
Optimized Document Ingestion Pipeline
Uses Qdrant for 10x faster vector storage and retrieval
UPDATED: Enhanced image-to-page linking with metadata
FIXED: Backward compatibility for PDF_OUTPUT_DIR
"""

import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import PictureItem, TableItem, DoclingDocument
import config
import json

# Backward compatibility: Create PDF_OUTPUT_DIR if not in config
if not hasattr(config, 'PDF_OUTPUT_DIR'):
    config.PDF_OUTPUT_DIR = config.STATIC_DIR / "pdfs"
    print("⚠️  PDF_OUTPUT_DIR not in config, using default: static/pdfs")

def setup_directories():
    """Setup output directories"""
    # Clean old storage if exists
    if config.STORAGE_DIR.exists():
        try:
            shutil.rmtree(config.STORAGE_DIR)
            print("🗑️  Removed old storage directory")
        except Exception as e:
            print(f"⚠️  Could not delete storage dir: {e}")
    
    # Create directories
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directories ready")

def setup_models():
    """Initialize embedding and LLM models"""
    print("\n⚙️  Initializing AI Models...")
    
    # Embedding model (HuggingFace - faster than Ollama)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_NAME,
        cache_folder=str(config.BASE_DIR / ".cache")
    )
    
    # LLM (only used for testing, not during ingestion)
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        request_timeout=config.LLM_REQUEST_TIMEOUT
    )
    
    # Chunking settings
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    
    print(f"✅ Embedding model: {config.EMBED_MODEL_NAME}")
    print(f"✅ Chunk size: {config.CHUNK_SIZE}")

def setup_docling():
    """Configure Docling for PDF processing"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = config.DOCLING_OCR
    pipeline_options.do_table_structure = config.DOCLING_TABLE_EXTRACTION
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = config.DOCLING_GENERATE_PICTURES
    pipeline_options.images_scale = config.DOCLING_IMAGE_SCALE
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    return converter

def extract_images_with_metadata(conv_result, doc_filename: str) -> dict:
    """
    Extract images with detailed page and position metadata
    
    Returns:
        dict: {image_filename: metadata_dict}
    """
    image_metadata = {}
    page_image_counts = {}
    
    for element, _level in conv_result.document.iterate_items():
        if isinstance(element, PictureItem):
            # Get page number
            page_num = None
            if hasattr(element, 'prov') and element.prov:
                for prov_item in element.prov:
                    if hasattr(prov_item, 'page_no'):
                        page_num = prov_item.page_no
                        break
            
            if page_num is None:
                continue  # Skip images without page info
            
            # Track images per page
            if page_num not in page_image_counts:
                page_image_counts[page_num] = 0
            page_image_counts[page_num] += 1
            fig_num = page_image_counts[page_num]
            
            # Generate filename: {doc}_page{X}_fig{Y}.png
            image_name = f"{doc_filename}_page{page_num}_fig{fig_num}.png"
            image_path = config.IMAGE_OUTPUT_DIR / image_name
            
            # Save image
            img_data = element.get_image(conv_result.document)
            if img_data:
                with open(image_path, "wb") as f:
                    img_data.save(f, "PNG")
                
                # Store metadata
                image_metadata[image_name] = {
                    "page": page_num,
                    "figure_num": fig_num,
                    "document": doc_filename,
                    "path": str(image_path)
                }
    
    return image_metadata

def build_page_to_text_mapping(docling_doc: DoclingDocument) -> dict:
    """
    Build a mapping of page numbers to their text content
    
    Returns:
        dict: {page_num: text_content}
    """
    page_text_map = {}
    
    # Iterate through document structure
    for item, level in docling_doc.iterate_items():
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    page_num = prov.page_no
                    
                    # Get text from item - skip PictureItem
                    if isinstance(item, PictureItem):
                        continue  # Skip images, we only want text
                    
                    if hasattr(item, 'text') and item.text:
                        text = item.text
                    elif hasattr(item, 'export_to_markdown'):
                        try:
                            text = item.export_to_markdown(docling_doc)
                        except:
                            continue
                    else:
                        continue
                    
                    if text and text.strip():  # Only add non-empty text
                        if page_num not in page_text_map:
                            page_text_map[page_num] = []
                        page_text_map[page_num].append(text)
    
    # Join texts for each page
    return {page: "\n".join(texts) for page, texts in page_text_map.items()}

def process_single_pdf(pdf_path: Path, converter: DocumentConverter) -> tuple:
    """
    Process a single PDF and extract text + images with page linking
    
    Returns:
        (list[Document], image_metadata, page_text_map)
    """
    print(f"   📄 Processing: {pdf_path.name}")
    start = time.time()
    
    # Copy PDF to static directory for serving
    pdf_static_path = config.PDF_OUTPUT_DIR / pdf_path.name
    try:
        shutil.copy2(pdf_path, pdf_static_path)
        print(f"      📋 PDF copied to static directory")
    except Exception as e:
        print(f"      ⚠️  Could not copy PDF: {e}")
    
    # Convert PDF
    conv_result = converter.convert(pdf_path)
    doc_filename = pdf_path.stem
    
    # Extract images with metadata
    image_metadata = extract_images_with_metadata(conv_result, doc_filename)
    print(f"      ✅ {len(image_metadata)} images extracted with page metadata")
    
    # Build page-to-text mapping
    page_text_map = build_page_to_text_mapping(conv_result.document)
    print(f"      ✅ {len(page_text_map)} pages mapped")
    
    # Export full markdown
    md_text = conv_result.document.export_to_markdown()
    
    # Create documents with enhanced page metadata
    # We'll chunk by page for better page attribution
    documents = []
    
    # Parse by pages if possible
    for page_num, page_text in page_text_map.items():
        if page_text.strip():  # Skip empty pages
            doc = Document(
                text=page_text,
                metadata={
                    "file_name": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page_label": str(page_num),
                    "page_number": page_num,  # Numeric for filtering
                    "document_stem": doc_filename
                }
            )
            documents.append(doc)
    
    # If page parsing failed, fall back to full document
    if not documents:
        documents = [Document(
            text=md_text,
            metadata={
                "file_name": pdf_path.name,
                "source_path": str(pdf_path),
                "document_stem": doc_filename
            }
        )]
    
    elapsed = time.time() - start
    print(f"      ⏱️  Completed in {elapsed:.1f}s")
    
    return documents, image_metadata, page_text_map

def run_ingestion():
    """Main ingestion pipeline"""
    print("\n" + "="*60)
    print("🚀 RPMG DOCUMENT INGESTION PIPELINE v3.0")
    print("   Enhanced Image-Context Linking")
    print("="*60)
    print(f"📂 Source: {config.DATA_DIR}")
    print(f"💾 Vector DB: Qdrant at {config.QDRANT_PATH}")
    print(f"🖼️  Images: {config.IMAGE_OUTPUT_DIR}")
    print(f"📋 PDFs: {config.PDF_OUTPUT_DIR}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    # Step 1: Setup
    setup_directories()
    setup_models()
    converter = setup_docling()
    
    # Step 2: Find PDFs
    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {config.DATA_DIR}")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF(s)")
    
    # Step 3: Process PDFs
    all_documents = []
    all_image_metadata = {}
    all_page_maps = {}
    
    for pdf_path in pdf_files:
        docs, img_meta, page_map = process_single_pdf(pdf_path, converter)
        all_documents.extend(docs)
        all_image_metadata.update(img_meta)
        all_page_maps[pdf_path.stem] = page_map
    
    # Save image metadata for quick lookup
    metadata_file = config.STORAGE_DIR / "image_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(all_image_metadata, f, indent=2)
    print(f"\n💾 Saved image metadata to {metadata_file}")
    
    # Step 4: Create Qdrant Vector Store
    print(f"\n🧠 Creating Qdrant vector database...")
    
    client = QdrantClient(path=str(config.QDRANT_PATH))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="piping_docs"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Step 5: Build Index with custom chunking to preserve page metadata
    print(f"   Embedding {len(all_documents)} document(s)...")
    embedding_start = time.time()
    
    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    embedding_time = time.time() - embedding_start
    total_time = time.time() - start_time
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("✅ INGESTION COMPLETE!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   PDFs processed: {len(pdf_files)}")
    print(f"   Documents created: {len(all_documents)}")
    print(f"   Images extracted: {len(all_image_metadata)}")
    if all_image_metadata:
        example_img = list(all_image_metadata.keys())[0]
        print(f"   Image naming: {example_img}")
    print(f"   Embedding time: {embedding_time:.1f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"\n💾 Vector database: {config.QDRANT_PATH}")
    print(f"🖼️  Images: {config.IMAGE_OUTPUT_DIR}")
    print(f"📋 PDFs: {config.PDF_OUTPUT_DIR}")
    print(f"📄 Metadata: {metadata_file}")
    print("\n✨ NEW FEATURES:")
    print("   ✅ Images linked to specific pages")
    print("   ✅ Page-level document chunking")
    print("   ✅ Source PDFs available for viewing")
    print("\n⚡ Next steps:")
    print("   1. Restart your FastAPI server: python main.py")
    print("   2. Images will only show for relevant context pages!")
    print("   3. Users can click to view exact PDF pages!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_ingestion()