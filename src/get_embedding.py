import numpy as np
from tqdm.auto import tqdm
import faiss
import pickle
import torch
import json
from typing import List, Dict

from resource_manager import ModelManager, cuda_memory_manager

model_manager = ModelManager()

def load_nq_dataset(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def process_nq_data(nq_data: Dict) -> List[Dict]:
    processed_data = []
    doc_id = 0
    
    for item in tqdm(nq_data, desc="Processing NQ data"):
        for ctx_type in ['positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs']:
            for chunk_index, ctx in enumerate(item[ctx_type]):
                doc_id += 1
                processed_data.append({
                    "text": ctx['text'].replace("\n", " ").strip(),
                    "title": ctx['title'],
                    "doc_id": f"nq_doc_{doc_id}",
                    "chunk_id": f"nq_doc_{doc_id}_chunk_{chunk_index}",
                    "length": len(ctx['text']),
                    "chunk_index": chunk_index,
                    "source": "natural_questions",
                    "model_version": model_manager.context_encoder.config.model_type
                })
    
    return processed_data

def process_batch(batch_items: List[Dict]) -> List[Dict]:
    inputs = model_manager.context_tokenizer(
        [item['text'] for item in batch_items],
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model_manager.device)
    
    with cuda_memory_manager():
        with torch.no_grad():
            embeddings = model_manager.context_encoder(**inputs).pooler_output
            embeddings = embeddings.cpu().numpy()
    
    return [
        {**item, "embedding": embedding}
        for item, embedding in zip(batch_items, embeddings)
    ]

def create_chunks_from_nq(processed_data: List[Dict], batch_size: int = 32) -> List[Dict]:
    chunks = []
    total_batches = (len(processed_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(processed_data), batch_size), desc="Creating chunks and embeddings", total=total_batches):
        chunks.extend(process_batch(processed_data[i:i + batch_size]))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return chunks

def build_and_save_faiss_index(chunks: List[Dict], index_path: str = "src/data/nq_faiss_index"):
    embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    
    metadata = [{
        'doc_id': chunk['doc_id'],
        'chunk_id': chunk['chunk_id'],
        'text': chunk['text'],
        'title': chunk['title'],
        'length': chunk['length'],
        'chunk_index': chunk['chunk_index'],
        'source': chunk['source'],
        'model_version': chunk['model_version']
    } for chunk in chunks]
    
    with open(f"{index_path}_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

def save_processed_data(chunks: List[Dict], output_file: str = "src/data/nq_trn_processed_data.json"):
    processed_data = [{
        'doc_id': chunk['doc_id'],
        'chunk_id': chunk['chunk_id'],
        'text': chunk['text'],
        'title': chunk['title'],
        'length': chunk['length'],
        'chunk_index': chunk['chunk_index'],
        'source': chunk['source'],
        'model_version': chunk['model_version'],
        'embedding': chunk['embedding'].tolist()
    } for chunk in chunks]
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)

if __name__ == "__main__":
    try:
        BATCH_SIZE = 16
        nq_file_path = "src/data/biencoder-nq-train.json"
        
        print(f"Starting process with batch size: {BATCH_SIZE}")
        nq_data = load_nq_dataset(nq_file_path)
        processed_data = process_nq_data(nq_data)
        print(f"Total items to process: {len(processed_data)}")
        
        chunks = create_chunks_from_nq(processed_data, batch_size=BATCH_SIZE)
        build_and_save_faiss_index(chunks)
        save_processed_data(chunks)
        
        print("\nProcess completed. FAISS index and JSON file saved.")
    finally:
        model_manager.cleanup()