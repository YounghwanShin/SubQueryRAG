import asyncio
import time
from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
import pandas as pd
import numpy as np

def load_processed_data(file_path: str) -> List[Dict[str, Any]]:
    text_chunks_and_embedding_df = pd.read_csv(file_path)
    
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))    

    processed_data = text_chunks_and_embedding_df.to_dict(orient="records")
    
    return processed_data

async def chatting(query: str) -> str:
    total_start_time = time.time()
    
    processed_data = load_processed_data("text_chunks_and_embeddings_df.csv")
    
    answer = await get_gpt_answer(query, processed_data)
    
    print(f"출력 시간: {time.time() - total_start_time:.2f} seconds")
    return answer

async def main():
    query = input("입력: ")
    result = await chatting(query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

# Can you explain the role of vitamin D in bone health and immune function, describe how the body processes different types of carbohydrates, and discuss the relationship between protein intake and muscle growth? Also, could you outline the main factors affecting iron absorption in the digestive system?