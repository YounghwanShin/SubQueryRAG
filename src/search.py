from typing import List, Dict, Any
import numpy as np
import faiss
import asyncio
from utils import get_query_embedding, RELEVANCE_THRESHOLD
import time
import pickle
from resource_manager import FAISSIndexManager, cuda_memory_manager
from query_processor import divide_query

faiss_manager = FAISSIndexManager()

async def search_query(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if faiss_manager.index is None:
        print("Warning: FAISS index is None in search_query")
        print(f"Manager instance id: {id(faiss_manager)}")
        raise RuntimeError("FAISS index not initialized properly")

    with cuda_memory_manager():
        query_embedding = await asyncio.to_thread(
            lambda: np.array([get_query_embedding(query)]).astype('float32')
        )
        faiss.normalize_L2(query_embedding)
        
        scores, indices = await asyncio.to_thread(
            lambda: faiss_manager.index.search(query_embedding, k)
        )
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= RELEVANCE_THRESHOLD:
            chunk_info = faiss_manager.chunk_data[idx].copy()
            chunk_info['relevance_score'] = float(score)
            results.append(chunk_info)
    
    return results

async def async_search_with_divide(original_query: str, k: int = 5) -> List[Dict[str, Any]]:
    sub_queries = await divide_query(original_query)
    
    search_tasks = [
        search_query(query, k)
        for query in sub_queries
    ]
    
    all_results = await asyncio.gather(*search_tasks)
    
    seen_chunks = set()
    merged_results = []
    
    for results in all_results:
        for result in results:
            if result['chunk_id'] not in seen_chunks:
                seen_chunks.add(result['chunk_id'])
                merged_results.append(result)
    
    merged_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return merged_results[:k]

async def test_search(query: str):
    try:
        index_path = "src/data/nq_faiss_index"
        print(f"Loading index from: {index_path}")
        faiss_manager.load_index(index_path)
        
        results = await search_query(query)
            
        print("\nSearch Results:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {result['relevance_score']:.4f}")
            print(f"Title: {result['title']}")
            print(f"Text: {result['text'][:200]}...")
            print("-" * 80)
            
    finally:
        faiss_manager.cleanup()

if __name__ == "__main__":
    query = "who was the first woman to win a Nobel Prize"
    asyncio.run(test_search(query))