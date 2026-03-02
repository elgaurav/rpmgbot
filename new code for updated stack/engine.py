"""
Optimized RAG Engine for RPMG Assistant
UPDATED: Smart query classification + support for larger models
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, List, Generator, Optional, Set
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import config

# ==================== GLOBAL INITIALIZATION ====================
print("🚀 Initializing RAG Engine...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=config.EMBED_MODEL_NAME,
    cache_folder=str(config.BASE_DIR / ".cache")
)

# UPGRADED: Larger context window for llama3.1:8b
# Increased num_ctx to support more retrieved context and longer answers
Settings.llm = Ollama(
    model=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    temperature=config.LLM_TEMPERATURE,
    additional_kwargs={
        "num_predict": config.LLM_MAX_TOKENS,
        "num_ctx": config.LLM_CONTEXT_SIZE,  # UPGRADED: 8k context window
    }
)

Settings.chunk_size = config.CHUNK_SIZE
Settings.chunk_overlap = config.CHUNK_OVERLAP

print(f"✅ Using embedding model: {config.EMBED_MODEL_NAME}")
print(f"✅ Using LLM: {config.LLM_MODEL} (context: {config.LLM_CONTEXT_SIZE} tokens)")

# ==================== QUERY CLASSIFICATION ====================

def is_casual_query(question: str) -> bool:
    """
    Determine if a query is casual (greeting/thanks) vs technical
    
    Returns:
        True if casual (don't need sources/images)
        False if technical (needs RAG retrieval)
    """
    if not config.ENABLE_QUERY_CLASSIFICATION:
        return False  # Always use RAG if classification disabled
    
    question_lower = question.lower().strip()
    
    # Check against casual patterns
    for pattern in config.CASUAL_QUERY_PATTERNS:
        if pattern in question_lower:
            # Exact match or at start/end
            if (question_lower == pattern or 
                question_lower.startswith(pattern + " ") or
                question_lower.endswith(" " + pattern) or
                question_lower.startswith(pattern + "!") or
                question_lower.startswith(pattern + "?")):
                return True
    
    # Very short queries (< 15 chars) without technical keywords
    technical_keywords = [
        "pipe", "piping", "stress", "plot", "plan", "API", "ASME",
        "pressure", "temperature", "flange", "valve", "material",
        "corrosion", "thickness", "design", "analysis", "standard",
        "code", "specification", "weld", "joint", "support"
    ]
    
    if len(question_lower) < 15:
        has_technical = any(kw.lower() in question_lower for kw in technical_keywords)
        if not has_technical:
            return True  # Short query without technical terms = casual
    
    return False

# ==================== SINGLETON PATTERN FOR QDRANT ====================

class QdrantManager:
    """Singleton manager for Qdrant client"""
    _instance: Optional['QdrantManager'] = None
    _client: Optional[QdrantClient] = None
    _index: Optional[VectorStoreIndex] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self) -> QdrantClient:
        if self._client is None:
            print("📦 Creating Qdrant client...")
            self._client = QdrantClient(path=str(config.QDRANT_PATH))
            print("✅ Qdrant client ready")
        return self._client
    
    def get_index(self) -> VectorStoreIndex:
        if self._index is None:
            print("📦 Loading Qdrant index...")
            client = self.get_client()
            
            collections = client.get_collections().collections
            collection_exists = any(c.name == "piping_docs" for c in collections)
            
            if not collection_exists:
                raise FileNotFoundError(
                    "No index found! Run ingest_pro.py first."
                )
            
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="piping_docs"
            )
            self._index = VectorStoreIndex.from_vector_store(vector_store)
            print("✅ Index loaded from Qdrant")
        
        return self._index
    
    def clear(self):
        if self._client is not None:
            try:
                self._client.close()
            except:
                pass
        self._client = None
        self._index = None
        print("🔄 Qdrant cache cleared")

_qdrant_manager = QdrantManager()

# ==================== IMAGE METADATA ====================

_IMAGE_METADATA_CACHE = None

def _load_image_metadata() -> dict:
    global _IMAGE_METADATA_CACHE
    
    if _IMAGE_METADATA_CACHE is None:
        metadata_file = config.STORAGE_DIR / "image_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                _IMAGE_METADATA_CACHE = json.load(f)
            print(f"📸 Loaded metadata for {len(_IMAGE_METADATA_CACHE)} images")
        else:
            _IMAGE_METADATA_CACHE = {}
    
    return _IMAGE_METADATA_CACHE

def _is_valid_page(page_label) -> bool:
    """Check if page label is valid"""
    if not page_label:
        return False
    
    page_str = str(page_label).strip().upper()
    invalid_patterns = ['N/A', 'N.A.', 'NONE', 'NULL', '']
    
    if page_str in invalid_patterns:
        return False
    
    try:
        int(page_str)
        return True
    except ValueError:
        return len(page_str) > 0 and page_str not in invalid_patterns

def _get_images_for_pages(document_stem: str, page_numbers: Set[int]) -> List[str]:
    """Get images for specific pages"""
    metadata = _load_image_metadata()
    matched_images = []
    
    expanded_pages = set()
    for page in page_numbers:
        expanded_pages.add(page)
        for offset in range(-config.IMAGE_ADJACENT_PAGES, config.IMAGE_ADJACENT_PAGES + 1):
            if offset != 0:
                expanded_pages.add(page + offset)
    
    for img_name, img_meta in metadata.items():
        if img_meta['document'] == document_stem and img_meta['page'] in expanded_pages:
            matched_images.append(img_name)
    
    if len(matched_images) > config.MAX_IMAGES_PER_QUERY:
        matched_images = matched_images[:config.MAX_IMAGES_PER_QUERY]
    
    return sorted(matched_images)

# ==================== QUERY FUNCTIONS ====================

def query_piping_data(question: str, stream: bool = False) -> Dict:
    """
    Query with smart classification
    
    IMPORTANT: For casual queries, returns direct LLM response without sources/images
    """
    start_time = time.time()
    
    try:
        # STEP 1: Classify the query
        if is_casual_query(question):
            # Casual query - just use LLM directly, no RAG
            if config.VERBOSE:
                print(f"💬 Casual query detected: '{question}' - skipping RAG")
            
            llm = Settings.llm
            response = llm.complete(question)
            total_time = time.time() - start_time
            
            return {
                "answer": str(response),
                "sources": [],  # NO SOURCES for casual queries
                "images": [],   # NO IMAGES for casual queries
                "timing": {
                    "total_seconds": round(total_time, 2),
                    "retrieval_seconds": 0.0,
                    "llm_seconds": round(total_time, 2)
                },
                "query_type": "casual"
            }
        
        # STEP 2: Technical query - use RAG
        if config.VERBOSE:
            print(f"🔍 Technical query detected: '{question}' - using RAG")
        
        index = _qdrant_manager.get_index()
        
        query_engine = index.as_query_engine(
            similarity_top_k=config.SIMILARITY_TOP_K,
            streaming=stream,
            verbose=config.VERBOSE
        )
        
        retrieval_start = time.time()
        response = query_engine.query(question)
        retrieval_time = time.time() - retrieval_start
        
        # Extract sources with validation
        sources = []
        page_map = {}
        
        for node in response.source_nodes:
            file_name = node.metadata.get("file_name", "Unknown")
            page_label = node.metadata.get("page_label", None)
            page_number = node.metadata.get("page_number", None)
            doc_stem = node.metadata.get("document_stem", Path(file_name).stem)
            score = node.score if hasattr(node, 'score') else None
            
            # CRITICAL: Skip invalid pages AND low-relevance matches
            if not _is_valid_page(page_label) or not page_number:
                continue
            
            if score and score < config.MIN_RELEVANCE_SCORE:
                if config.VERBOSE:
                    print(f"⚠️  Skipping low-relevance source (score: {score:.3f})")
                continue
            
            # Build PDF link
            pdf_link = None
            if file_name != "Unknown" and hasattr(config, 'PDF_OUTPUT_DIR'):
                if page_number:
                    pdf_link = f"{config.BASE_URL}/static/pdfs/{file_name}#page={page_number}"
                else:
                    pdf_link = f"{config.BASE_URL}/static/pdfs/{file_name}"
            
            sources.append({
                "file": file_name,
                "page": page_label,
                "score": round(score, 3) if score else None,
                "pdf_link": pdf_link,
                "page_number": page_number
            })
            
            # Track pages for images
            if page_number and doc_stem:
                if doc_stem not in page_map:
                    page_map[doc_stem] = set()
                page_map[doc_stem].add(page_number)
        
        # Get images ONLY if we have valid sources
        found_images = []
        if sources and config.IMAGE_PAGE_MATCH_STRICT:
            for doc_stem, pages in page_map.items():
                doc_images = _get_images_for_pages(doc_stem, pages)
                found_images.extend(doc_images)
        
        # If no valid sources, don't show images
        if not sources:
            found_images = []
        
        total_time = time.time() - start_time
        
        result = {
            "answer": str(response),
            "sources": sources,
            "images": sorted(list(set(found_images)))[:config.MAX_IMAGES_PER_QUERY] if found_images else [],
            "timing": {
                "total_seconds": round(total_time, 2),
                "retrieval_seconds": round(retrieval_time, 2),
                "llm_seconds": round(total_time - retrieval_time, 2)
            },
            "query_type": "technical"
        }
        
        if config.VERBOSE:
            print(f"⏱️  Query completed in {total_time:.2f}s")
            print(f"📊 Retrieved {len(sources)} valid chunks, {len(found_images)} images")
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.2f}s: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "images": [],
            "timing": {"total_seconds": round(error_time, 2)}
        }

def query_piping_data_stream(question: str) -> Generator[str, None, None]:
    """Stream response tokens"""
    try:
        # Check if casual query
        if is_casual_query(question):
            llm = Settings.llm
            response = llm.complete(question)
            yield str(response)
            return
        
        # Technical query with streaming
        index = _qdrant_manager.get_index()
        query_engine = index.as_query_engine(
            similarity_top_k=config.SIMILARITY_TOP_K,
            streaming=True
        )
        
        response = query_engine.query(question)
        
        for token in response.response_gen:
            yield token
            
    except Exception as e:
        yield f"\n\n❌ Error: {str(e)}"

def clear_cache():
    """Clear all caches"""
    global _IMAGE_METADATA_CACHE
    _qdrant_manager.clear()
    _IMAGE_METADATA_CACHE = None
    print("🔄 All caches cleared")

def get_stats() -> Dict:
    """Get system statistics"""
    try:
        index = _qdrant_manager.get_index()
        client = _qdrant_manager.get_client()
        collection_info = client.get_collection("piping_docs")
        metadata = _load_image_metadata()
        
        return {
            "status": "ready",
            "vector_count": collection_info.points_count,
            "embedding_model": config.EMBED_MODEL_NAME,
            "llm_model": config.LLM_MODEL,
            "context_size": config.LLM_CONTEXT_SIZE,
            "images_cached": len(metadata),
            "image_page_matching": config.IMAGE_PAGE_MATCH_STRICT,
            "query_classification": config.ENABLE_QUERY_CLASSIFICATION
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e)
        }

# Expose for main.py
_load_or_create_index = lambda: _qdrant_manager.get_index()
_get_image_cache = lambda: set(_load_image_metadata().keys())

def get_query_metadata(question: str) -> dict:
    """Get metadata without full answer generation"""
    
    # IMPORTANT: Also check if casual query here
    if is_casual_query(question):
        return {"sources": [], "images": []}
    
    try:
        index = _qdrant_manager.get_index()
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        nodes = retriever.retrieve(question)
        
        sources = []
        page_map = {}
        
        for node in nodes:
            file_name = node.metadata.get("file_name", "Unknown")
            page_label = node.metadata.get("page_label", None)
            page_number = node.metadata.get("page_number", None)
            doc_stem = node.metadata.get("document_stem", Path(file_name).stem)
            score = node.score if hasattr(node, 'score') else None
            
            if not _is_valid_page(page_label) or not page_number:
                continue
            
            if score and score < config.MIN_RELEVANCE_SCORE:
                continue
            
            pdf_link = None
            if file_name != "Unknown" and hasattr(config, 'PDF_OUTPUT_DIR'):
                if page_number:
                    pdf_link = f"{config.BASE_URL}/static/pdfs/{file_name}#page={page_number}"
            
            sources.append({
                "file": file_name,
                "page": page_label,
                "pdf_link": pdf_link
            })
            
            if page_number and doc_stem:
                if doc_stem not in page_map:
                    page_map[doc_stem] = set()
                page_map[doc_stem].add(page_number)
        
        found_images = []
        if sources:
            for doc_stem, pages in page_map.items():
                doc_images = _get_images_for_pages(doc_stem, pages)
                found_images.extend(doc_images)
        
        image_urls = [
            f"{config.BASE_URL}/static/images/{img}" 
            for img in sorted(set(found_images))[:config.MAX_IMAGES_PER_QUERY]
        ] if found_images else []
        
        return {
            "sources": sources,
            "images": image_urls
        }
        
    except Exception as e:
        print(f"❌ Metadata error: {e}")
        return {"sources": [], "images": []}
