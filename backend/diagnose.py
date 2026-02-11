#!/usr/bin/env python3
"""
Performance Diagnostic Tool for RPMG RAG Assistant

This script tests each component to identify bottlenecks:
1. Ollama LLM performance
2. Embedding performance  
3. Vector store retrieval speed
4. Overall RAG pipeline
5. NEW: Image-page linking validation
6. NEW: PDF source link generation
"""

import time
import sys
from pathlib import Path

def test_ollama_speed():
    """Test Ollama model inference speed"""
    print("\n" + "="*60)
    print("TEST 1: Ollama LLM Performance")
    print("="*60)
    
    try:
        from llama_index.llms.ollama import Ollama
        import config
        
        llm = Ollama(
            model=config.LLM_MODEL,
            request_timeout=60.0
        )
        
        test_prompt = "Explain corrosion in exactly 50 words."
        
        print(f"Model: {config.LLM_MODEL}")
        print(f"Prompt: '{test_prompt}'")
        print("Generating response...")
        
        start = time.time()
        response = llm.complete(test_prompt)
        elapsed = time.time() - start
        
        print(f"\n✅ Response generated in {elapsed:.2f}s")
        print(f"Response length: {len(str(response))} chars")
        print(f"Response preview: {str(response)[:200]}...")
        
        # Evaluate speed
        if elapsed < 10:
            print(f"🚀 EXCELLENT - <10s response time")
        elif elapsed < 30:
            print(f"✅ GOOD - <30s response time")
        elif elapsed < 60:
            print(f"⚠️  SLOW - Consider upgrading to llama3.1:8b")
        else:
            print(f"❌ VERY SLOW - Model is too large for your hardware")
            print(f"   Recommendation: Use llama3.2:3b instead")
        
        return elapsed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_embedding_speed():
    """Test embedding generation speed"""
    print("\n" + "="*60)
    print("TEST 2: Embedding Performance")
    print("="*60)
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import config
        
        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            cache_folder=str(config.BASE_DIR / ".cache")
        )
        
        test_text = "This is a test document for embedding performance." * 10
        
        print(f"Model: {config.EMBED_MODEL_NAME}")
        print(f"Text length: {len(test_text)} chars")
        print("Generating embeddings...")
        
        # Warmup
        _ = embed_model.get_text_embedding(test_text)
        
        # Actual test
        start = time.time()
        for _ in range(10):
            embedding = embed_model.get_text_embedding(test_text)
        elapsed = time.time() - start
        avg_time = elapsed / 10
        
        print(f"\n✅ Average embedding time: {avg_time*1000:.1f}ms")
        print(f"Embedding dimension: {len(embedding)}")
        
        if avg_time < 0.1:
            print("🚀 EXCELLENT - Very fast embeddings")
        elif avg_time < 0.5:
            print("✅ GOOD - Acceptable speed")
        else:
            print("⚠️  SLOW - Consider using a smaller embedding model")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_vector_store_speed():
    """Test vector store retrieval speed"""
    print("\n" + "="*60)
    print("TEST 3: Vector Store Retrieval")
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
            
            # Test search speed
            test_query = [0.1] * collection_info.config.params.vectors.size
            
            print("Running 10 search queries...")
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
            print(f"Top-K: {config.SIMILARITY_TOP_K}")
            
            if avg_time < 0.05:
                print("🚀 EXCELLENT - Very fast retrieval")
            elif avg_time < 0.2:
                print("✅ GOOD - Acceptable speed")
            else:
                print("⚠️  SLOW - Large index or slow disk")
            
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
    """Test full RAG pipeline with new features"""
    print("\n" + "="*60)
    print("TEST 6: Full RAG Pipeline (with source links)")
    print("="*60)
    
    try:
        from engine import query_piping_data
        
        test_question = "What is corrosion?"
        
        print(f"Question: '{test_question}'")
        print("Running full RAG query...")
        
        start = time.time()
        result = query_piping_data(test_question)
        elapsed = time.time() - start
        
        print(f"\n✅ Query completed in {elapsed:.2f}s")
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Sources found: {len(result['sources'])}")
        print(f"Images found: {len(result['images'])}")
        
        # Show source details with PDF links
        print(f"\n📄 Sources with links:")
        for i, source in enumerate(result['sources'][:3], 1):
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
        
        # Evaluate
        if elapsed < 10:
            print(f"\n🚀 EXCELLENT - Well under 30s target!")
        elif elapsed < 30:
            print(f"\n✅ SUCCESS - Met <30s target!")
        elif elapsed < 60:
            print(f"\n⚠️  BORDERLINE - Close to 1 minute")
            print("   Consider upgrading your LLM model")
        else:
            print(f"\n❌ FAILED - Exceeds 30s target")
            print("   Primary bottleneck is likely the LLM")
            print("   Recommendations:")
            print("   1. Upgrade to llama3.2:3b minimum")
            print("   2. Use llama3.1:8b if you have 8GB+ RAM")
            print("   3. Enable GPU acceleration if available")
        
        return elapsed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you've run ingest_pro.py first")
        return None

def main():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("🔍 RPMG RAG ASSISTANT - PERFORMANCE DIAGNOSTICS v3.0")
    print("="*60)
    
    # Import config
    try:
        import config
        print(f"\n📊 Current Configuration:")
        print(f"   LLM Model: {config.LLM_MODEL}")
        print(f"   Embedding Model: {config.EMBED_MODEL_NAME}")
        print(f"   Top-K Retrieval: {config.SIMILARITY_TOP_K}")
        print(f"   Max Tokens: {config.LLM_MAX_TOKENS}")
        print(f"   Chunk Size: {config.CHUNK_SIZE}")
        print(f"\n🖼️  Image Configuration:")
        print(f"   Strict page matching: {config.IMAGE_PAGE_MATCH_STRICT}")
        print(f"   Adjacent pages: ±{config.IMAGE_ADJACENT_PAGES}")
        print(f"   Max images per query: {config.MAX_IMAGES_PER_QUERY}")
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
        print(f"Ollama LLM: {results['ollama']:.2f}s")
    if results['embedding']:
        print(f"Embeddings: {results['embedding']*1000:.1f}ms avg")
    if results['vector_store']:
        print(f"Vector Search: {results['vector_store']*1000:.1f}ms avg")
    if results['image_metadata']:
        print(f"Image Metadata: {results['image_metadata']} images indexed")
    if results['pdf_links']:
        print(f"Source PDFs: {results['pdf_links']} available")
    if results['pipeline']:
        print(f"Full Pipeline: {results['pipeline']:.2f}s")
        
        if results['pipeline'] < 30:
            print("\n🎉 CONGRATULATIONS! Your system meets the <30s target!")
        else:
            print("\n❌ System exceeds 30s target.")
            print("\n🔧 RECOMMENDATIONS:")
            if results['ollama'] and results['ollama'] > 20:
                print("   1. PRIMARY ISSUE: LLM is too slow")
                print(f"      Current: {config.LLM_MODEL}")
                print("      Try: llama3.2:3b (minimum) or llama3.1:8b (recommended)")
            print("   2. Reduce LLM_MAX_TOKENS in config.py")
            print("   3. Reduce SIMILARITY_TOP_K to 1 for faster retrieval")
            print("   4. Consider GPU acceleration")
    
    print("\n✨ NEW FEATURES STATUS:")
    if results['image_metadata']:
        print("   ✅ Image-page linking: WORKING")
    else:
        print("   ❌ Image-page linking: NOT CONFIGURED")
    
    if results['pdf_links']:
        print("   ✅ PDF source links: WORKING")
    else:
        print("   ❌ PDF source links: NOT CONFIGURED")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()