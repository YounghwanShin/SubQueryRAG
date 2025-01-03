import asyncio
import time
from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
import os
from resource_manager import ModelManager, FAISSIndexManager
import transformers
transformers.logging.set_verbosity_error()

model_manager = ModelManager()
faiss_manager = FAISSIndexManager()

async def initialize_search_engine():
    index_path = "src/data/nq_faiss_index"
    print(f"Looking for index at: {os.path.abspath(index_path)}")
    print(f"Main manager instance id: {id(faiss_manager)}")
    
    if os.path.exists(index_path):
        print("Found FAISS index file")
        faiss_manager.load_index(index_path)
        if faiss_manager.index is None:
            raise RuntimeError("Failed to load FAISS index")
        print("Successfully loaded FAISS index")
        print(f"Index dimension: {faiss_manager.index.d}")
        print(f"Index total vectors: {faiss_manager.index.ntotal}")
    else:
        raise FileNotFoundError(f"FAISS index not found at {os.path.abspath(index_path)}. Run get_embedding.py first.")

async def chatting(query: str) -> str:
    total_start_time = time.time()
    answer = await get_gpt_answer(query)
    print(f"출력 시간: {time.time() - total_start_time:.2f} seconds")
    return answer

async def main():
    try:
        await initialize_search_engine()
        while True:
            query = input("입력: ")
            if query.lower() == 'quit':
                break
            result = await chatting(query)
            print(result)
    finally:
        model_manager.cleanup()
        faiss_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

# Can you explain the role of vitamin D in bone health and immune function, describe how the body processes different types of carbohydrates, and discuss the relationship between protein intake and muscle growth? Also, could you outline the main factors affecting iron absorption in the digestive system?