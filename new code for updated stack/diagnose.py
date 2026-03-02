#!/usr/bin/env python3
"""
Performance Diagnostic Tool for RPMG RAG Assistant
UPDATED: Tests for larger models (bge-large, llama3.1:8b)

This script tests each component to identify bottlenecks:
1. Ollama LLM performance (llama3.1:8b)
2. Embedding performance (bge-large)
3. Vector store retrieval speed
4. Overall RAG pipeline
5. Image-page linking validation
6. PDF source link generation
"""

import time
import sys
from pathlib import Path

def test_ollama_speed():
    """Test Ollama model inference speed with larger model"""
    print("\n" + "="*60)
    print("TEST 1: Ollama LLM Performance (llama3.1:8b)")
    print("="*60)
    
    try:
        from llama_index.llms.ollama import Ollama
        import config
        
        llm = Ollama(
            model=config.LLM_MODEL,
            request_timeout=config.LLM_REQUEST_TIMEOUT,
            additional_kwargs={
                "num_predict": config.LLM_MAX_TOKENS,
                "num_ctx": config.LLM_CONTEXT_SIZE,  # Test with larger context
            }
        )
        
        # More complex prompt to test larger model capabilities
        test_prompt = "Explain the key differences between API 5L and ASME B31.3 piping standards in exactly 100 words."
        
        print(f"Model: {config.LLM_MODEL}")
        print(f"Context Window: {config.LLM_CONTEXT_SIZE} tokens")
        print(f"Max Tokens: {config.LLM_MAX_TOKENS}")
        print(f"Prompt: '{test_prompt}'")
        print("Generating response...")
        
        start = time.time()
        response = llm.complete(test_prompt)
        elapsed = time.time() - start
        
        print(f"\n✅ Response generated in {elapsed:.2f}s")
        print(f"Response length: {len(str(response))} chars")
        print(f"Response preview: {str(response)[:200]}...")
        
        # Evaluate speed for larger model
        # llama3.1:8b is slower but much better quality than 1.5b models
        if elapsed < 15:
            print(f"🚀 EXCELLENT - <15s response time for 8B model")
        elif elapsed < 30:
            print(f"✅ GOOD - <30s response time")
        elif elapsed < 60:
            print(f"⚠️  ACCEPTABLE - Model is working but consider GPU acceleration")
        else:
            print(f"❌ SLOW - Consider enabling GPU or reducing model size")
            print(f"   Recommendation: Ensure Ollama is using GPU if available")
        
        return elapsed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_embedding_speed():
    """Test embedding generation speed with larger model"""
    print("\n" + "="*60)
    print("TEST 2: Embedding Performance (bge-large)")
    print("="*60)
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import config
        
        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            cache_folder=str(config.BASE_DIR / ".cache")
        )
        
        # Longer test text for larger chunks
        test_text = "This is a test document for embedding performance. " * 20  # ~200 words
        
        print(f"Model: {config.EMBED_MODEL_NAME}")
        print(f"Embedding Dimension: 1024 (vs 384 for bge-small)")
        print(f"Text length: {len(test_text)} chars")
        print("Generating embeddings...")
        
        # Warmup
        _ = embed_model.get_text_embedding(test_text)
        
        # Actual test with 10 iterations
        start = time.time()
        for _ in range(10):
            embedding = embed_model.get_text_embedding(test_text)
        elapsed = time.time() - start
        avg_time = elapsed / 10
        
        print(f"\n✅ Average embedding time: {avg_time*1000:.1f}ms")
        print(f"Embedding dimension: {len(embedding)}")
        
        # bge-large is slower but more accurate than bge-small
        if avg_time < 0.2:
            print("🚀 EXCELLENT - Very fast embeddings for large model")
        elif avg_time < 0.5:
            print("✅ GOOD - Acceptable speed")
        elif avg_time < 1.0:
            print("⚠️  ACCEPTABLE - Consider GPU acceleration")
        else:
            print("⚠️  SLOW - Enable GPU or reduce embedding model size")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_vector_store_speed():
    """Test vector store retrieval speed with larger vectors"""
    print("\n" + "="*60)
    print("TEST 3: Vector Store Retrieval (1024-dim vectors)")
    print("="*60)
    
    try:
        from qdrant_client import QdrantClient
        import config
        
        if not config.QDRANT_PATH.exists():
            print("⚠️  No index found. Run ingest_pro.py first.")
            return None
        
        client = QdrantClient(path=str(config.QDRANT_PATH))
        
        try:
            collection_info = client.get_collection("piping_docs")
            print(f"Collection: piping_docs")
            print(f"Vectors: {collection_info.points_count}")
            print(f"Vector Dimension: {collection_info.config.params.vectors.size}")
            
            # Test search speed with larger top_k
            test_query = [0.1] * collection_info.config.params.vectors.size
            
            print(f"Running 10 search queries with top_k={config.SIMILARITY_TOP_K}...")
            start = time.time()
            for _ in range(10):
                results = client.search(
                    collection_name="piping_docs",
                    query_vector=test_query,
                    limit=config.SIMILARITY_TOP_K
                )
            elapsed = time.time() - start
            avg_time = elapsed / 10
            
            print(f"\n✅ Average search time: {avg_time*1000:.1f}ms")
            print(f"Top-K: {config.SIMILARITY_TOP_K} (increased from 2)")
            
            if avg_time < 0.1:
                print("🚀 EXCELLENT - Very fast retrieval")
            elif avg_time < 0.3:
                print("✅ GOOD - Acceptable speed")
            else:
                print("⚠️  ACCEPTABLE - Large index or slower disk")
            
            return avg_time
            
        except Exception as e:
            print(f"❌ Collection not found: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_image_metadata():
    """Test image metadata and page linking"""
    print("\n" + "="*60)
    print("TEST 4: Image-Page Linking")
    print("="*60)
    
    try:
        import config
        import json
        
        metadata_file = config.STORAGE_DIR / "image_metadata.json"
        
        if not metadata_file.exists():
            print("⚠️  No image metadata found. Run ingest_pro.py first.")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded metadata for {len(metadata)} images")
        
        # Analyze metadata
        docs = {}
        for img_name, img_meta in metadata.items():
            doc = img_meta['document']
            if doc not in docs:
                docs[doc] = {'pages': set(), 'images': 0}
            docs[doc]['pages'].add(img_meta['page'])
            docs[doc]['images'] += 1
        
        print(f"\n📊 Documents with images:")
        for doc, info in docs.items():
            print(f"   {doc}:")
            print(f"      - {info['images']} images")
            print(f"      - Across {len(info['pages'])} pages")
            print(f"      - Pages: {sorted(list(info['pages']))[:5]}...")
        
        # Test image filtering logic
        from engine import _get_images_for_pages
        
        test_doc = list(docs.keys())[0]
        test_pages = set(list(docs[test_doc]['pages'])[:2])
        
        print(f"\n🧪 Testing image filter:")
        print(f"   Document: {test_doc}")
        print(f"   Pages: {test_pages}")
        
        filtered_images = _get_images_for_pages(test_doc, test_pages)
        print(f"   ✅ Found {len(filtered_images)} images for these pages")
        
        if config.IMAGE_PAGE_MATCH_STRICT:
            print(f"   ✅ Strict matching enabled - only exact pages")
        else:
            print(f"   ⚠️  Loose matching - may include extra images")
        
        return len(metadata)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_pdf_links():
    """Test PDF source links"""
    print("\n" + "="*60)
    print("TEST 5: PDF Source Links")
    print("="*60)
    
    try:
        import config
        
        if not config.PDF_OUTPUT_DIR.exists():
            print("⚠️  PDF output directory not found. Run ingest_pro.py first.")
            return None
        
        pdfs = list(config.PDF_OUTPUT_DIR.glob("*.pdf"))
        
        print(f"✅ Found {len(pdfs)} source PDFs")
        
        for pdf in pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   📄 {pdf.name} ({size_mb:.2f} MB)")
            
            # Generate example link with page anchor
            example_link = f"{config.BASE_URL}/static/pdfs/{pdf.name}#page=5"
            print(f"      Link example: {example_link}")
        
        if pdfs:
            print(f"\n✅ PDF serving ready at {config.BASE_URL}/static/pdfs/")
            print(f"   Links will include #page=N anchors for direct navigation")
        
        return len(pdfs)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_full_pipeline():
    """Test full RAG pipeline with larger models"""
    print("\n" + "="*60)
    print("TEST 6: Full RAG Pipeline (with upgraded models)")
    print("="*60)
    
    try:
        from engine import query_piping_data
        
        # More complex technical question to test larger model capabilities
        test_question = "What are the key considerations for pipe stress analysis in high-temperature applications?"
        
        print(f"Question: '{test_question}'")
        print("Running full RAG query with:")
        print(f"  - LLM: llama3.1:8b (context: 8k tokens)")
        print(f"  - Embeddings: bge-large (1024-dim)")
        print(f"  - Top-K: {config.SIMILARITY_TOP_K}")
        print(f"  - Chunk Size: {config.CHUNK_SIZE}")
        
        start = time.time()
        result = query_piping_data(test_question)
        elapsed = time.time() - start
        
        print(f"\n✅ Query completed in {elapsed:.2f}s")
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Sources found: {len(result['sources'])}")
        print(f"Images found: {len(result['images'])}")
        
        # Show source details with PDF links
        print(f"\n📄 Sources with links:")
        for i, source in enumerate(result['sources'][:4], 1):  # Show up to 4 sources
            print(f"   {i}. {source['file']} (page {source['page']})")
            if source.get('pdf_link'):
                print(f"      Link: {source['pdf_link']}")
            if source.get('score'):
                print(f"      Relevance: {source['score']}")
        
        # Show image details
        if result['images']:
            print(f"\n🖼️  Images (filtered by page context):")
            for img in result['images'][:3]:
                print(f"   - {img}")
        
        # Show timing breakdown if available
        if 'timing' in result:
            timing = result['timing']
            print(f"\n⏱️  Timing breakdown:")
            print(f"   - Retrieval: {timing.get('retrieval_seconds', 0):.2f}s")
            print(f"   - LLM: {timing.get('llm_seconds', 0):.2f}s")
            print(f"   - Total: {timing.get('total_seconds', 0):.2f}s")
        
        # Evaluate - larger models may take longer but provide better answers
        print(f"\n📊 Performance Assessment:")
        if elapsed < 20:
            print(f"🚀 EXCELLENT - Very fast for 8B model!")
        elif elapsed < 40:
            print(f"✅ GOOD - Acceptable performance")
            print("   Note: Larger models (llama3.1:8b) trade speed for quality")
        elif elapsed < 60:
            print(f"⚠️  ACCEPTABLE - Within 1 minute")
            print("   Consider GPU acceleration for faster inference")
        else:
            print(f"⚠️  SLOW - Exceeds 1 minute")
            print("   Recommendations:")
            print("   1. Enable GPU acceleration in Ollama")
            print("   2. Reduce LLM_MAX_TOKENS in config.py")
            print("   3. Ensure sufficient RAM (16GB+ recommended)")
        
        return elapsed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you've run ingest_pro.py first")
        return None

def main():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("🔍 RPMG RAG ASSISTANT - PERFORMANCE DIAGNOSTICS v4.0")
    print("   UPGRADED: Testing larger models (llama3.1:8b + bge-large)")
    print("="*60)
    
    # Import config
    try:
        import config
        print(f"\n📊 Current Configuration:")
        print(f"   LLM Model: {config.LLM_MODEL}")
        print(f"   Context Window: {config.LLM_CONTEXT_SIZE} tokens")
        print(f"   Max Tokens: {config.LLM_MAX_TOKENS}")
        print(f"   Embedding Model: {config.EMBED_MODEL_NAME}")
        print(f"   Top-K Retrieval: {config.SIMILARITY_TOP_K}")
        print(f"   Chunk Size: {config.CHUNK_SIZE}")
        print(f"   Chunk Overlap: {config.CHUNK_OVERLAP}")
        print(f"\n🖼️  Image Configuration:")
        print(f"   Strict page matching: {config.IMAGE_PAGE_MATCH_STRICT}")
        print(f"   Adjacent pages: ±{config.IMAGE_ADJACENT_PAGES}")
        print(f"   Max images per query: {config.MAX_IMAGES_PER_QUERY}")
        print(f"\n⚙️  Hardware:")
        print(f"   GPU Enabled: {config.USE_GPU}")
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return
    
    results = {}
    
    # Run tests
    results['ollama'] = test_ollama_speed()
    results['embedding'] = test_embedding_speed()
    results['vector_store'] = test_vector_store_speed()
    results['image_metadata'] = test_image_metadata()
    results['pdf_links'] = test_pdf_links()
    results['pipeline'] = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if results['ollama']:
        print(f"Ollama LLM (8B): {results['ollama']:.2f}s")
    if results['embedding']:
        print(f"Embeddings (1024-dim): {results['embedding']*1000:.1f}ms avg")
    if results['vector_store']:
        print(f"Vector Search (top-{config.SIMILARITY_TOP_K}): {results['vector_store']*1000:.1f}ms avg")
    if results['image_metadata']:
        print(f"Image Metadata: {results['image_metadata']} images indexed")
    if results['pdf_links']:
        print(f"Source PDFs: {results['pdf_links']} available")
    if results['pipeline']:
        print(f"Full Pipeline: {results['pipeline']:.2f}s")
        
        print(f"\n🎯 PERFORMANCE ANALYSIS:")
        if results['pipeline'] < 40:
            print("✅ System performance is good for upgraded models")
            print("   Larger models provide better answer quality")
        else:
            print("⚠️  System could be optimized")
            print("\n🔧 RECOMMENDATIONS:")
            if results['ollama'] and results['ollama'] > 30:
                print("   1. PRIMARY: Enable GPU acceleration in Ollama")
                print("      Run: ollama serve (with CUDA/Metal support)")
            if results['embedding'] and results['embedding'] > 0.5:
                print("   2. Consider GPU for embeddings (use device='cuda')")
            print("   3. Ensure at least 16GB RAM for smooth operation")
            print("   4. SSD recommended for vector store performance")
    
    print("\n✨ UPGRADED FEATURES:")
    print("   ✅ 8B parameter LLM (3x larger than before)")
    print("   ✅ 1024-dim embeddings (2.7x more dimensions)")
    print("   ✅ 8k context window (4x larger)")
    print("   ✅ 1024 token chunks (2x larger)")
    print("   ✅ Top-4 retrieval (2x more sources)")
    
    print("\n💡 NOTES:")
    print("   - Larger models trade some speed for significantly better quality")
    print("   - Expected response time: 15-40s (vs 10-30s for smaller models)")
    print("   - GPU acceleration highly recommended for optimal performance")
    print("   - Memory usage: ~8GB RAM + 4GB VRAM ideal")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
