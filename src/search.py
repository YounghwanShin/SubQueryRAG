from typing import List, Dict, Any
import numpy as np
import faiss
import asyncio
from utils import get_embedding, RELEVANCE_THRESHOLD
from QueryDivide import divide_query
import time

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.chunk_data = []
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        embeddings = []
        self.chunk_data = []
        
        for chunk in chunks:
            embeddings.append(chunk['embedding'])
            self.chunk_data.append({
                'page_number': chunk['page_number'],
                'sentence_chunk': chunk['sentence_chunk'],
                'chunk_char_count': chunk['chunk_char_count'],
                'chunk_word_count': chunk['chunk_word_count'],
                'chunk_token_count': chunk['chunk_token_count']
            })
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)  
        self.index.add(embeddings_array)
    
    async def async_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = await asyncio.to_thread(
            lambda: np.array([get_embedding(query)]).astype('float32')
        )
        faiss.normalize_L2(query_embedding)
        
        scores, indices = await asyncio.to_thread(
            lambda: self.index.search(query_embedding, k)
        )
        
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for score, idx in zip(scores, indices):
            if score >= RELEVANCE_THRESHOLD:
                chunk_info = self.chunk_data[idx]
                results.append({
                    **chunk_info,
                    'relevance_score': float(score)
                })
        
        return results

faiss_index = FAISSIndex()

async def async_search_with_divide(original_query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:    
    if faiss_index.index is None:
        faiss_index.build_index(chunks)
    
    sub_queries = await divide_query(original_query)
    
    search_tasks_start = time.time()
    search_tasks = [
        faiss_index.async_search(query, k)
        for query in sub_queries
    ]
    
    all_results = await asyncio.gather(*search_tasks)
    print(f"검색 시간 측정: {time.time() - search_tasks_start:.2f} seconds")
    
    seen_chunks = set()
    merged_results = []
    
    for results in all_results:
        for result in results:
            chunk_id = (result['page_number'], result['sentence_chunk'])
            
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                merged_results.append(result)
    
    merged_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return merged_results[:k]

def search(query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    return asyncio.run(async_search_with_divide(query, chunks, k))