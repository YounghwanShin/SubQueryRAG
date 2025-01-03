import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from spacy.lang.en import English
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import re
import faiss
import pickle
import torch
import json
from typing import List, Dict

from resource_manager import ModelManager, cuda_memory_manager

nlp = English()
nlp.add_pipe("sentencizer")
model_manager = ModelManager()

def text_formatter(text):
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def load_nq_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_nq_data(nq_data):
    processed_data = []
    
    for item in tqdm(nq_data, desc="Processing NQ data"):
        question = item['question']
        answers = item['answers']
        dataset = item['dataset']
        
        for ctx in item['positive_ctxs']:
            processed_data.append({
                "text": ctx['text'],
                "title": ctx['title'],
                "score": ctx['score'],
                "title_score": ctx['title_score'],
                "passage_id": ctx['passage_id']
            })
        
        for ctx in item['negative_ctxs']:
            processed_data.append({
                "text": ctx['text'],
                "title": ctx['title'],
                "score": ctx['score'],
                "title_score": ctx['title_score'],
                "passage_id": ctx['passage_id']
            })
            
        for ctx in item['hard_negative_ctxs']:
            processed_data.append({
                "text": ctx['text'],
                "title": ctx['title'],
                "score": ctx['score'],
                "title_score": ctx['title_score'],
                "passage_id": ctx['passage_id']
            })
    
    return processed_data

def process_batch(batch_items: List[Dict], batch_size: int = 32) -> List[Dict]:
    batch_texts = [text_formatter(item['text']) for item in batch_items]

    inputs = model_manager.context_tokenizer(
        batch_texts,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model_manager.device)
    
    with cuda_memory_manager():
        with torch.no_grad():
            embeddings = model_manager.context_encoder(**inputs).pooler_output
            embeddings = embeddings.cpu().numpy()
    
    processed_chunks = []
    for item, embedding in zip(batch_items, embeddings):
        chunk_dict = item.copy()
        chunk_dict["embedding"] = embedding
        processed_chunks.append(chunk_dict)
    
    return processed_chunks

def create_chunks_from_nq(processed_data, batch_size: int = 32):
    chunks = []
    total_batches = len(processed_data) // batch_size + (1 if len(processed_data) % batch_size != 0 else 0)
    
    for i in tqdm(range(0, len(processed_data), batch_size), desc="Creating chunks and embeddings", total=total_batches):
        batch_items = processed_data[i:i + batch_size]
        batch_chunks = process_batch(batch_items, batch_size)
        chunks.extend(batch_chunks)
    
    return chunks

def build_and_save_faiss_index(chunks, index_path="src/data/nq_faiss_index"):
    embeddings = [chunk['embedding'] for chunk in chunks]
    embeddings_array = np.array(embeddings).astype('float32')
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    faiss.write_index(index, index_path)
    
    metadata = []
    for chunk in chunks:
        meta = chunk.copy()
        del meta['embedding']  
        metadata.append(meta)
    
    with open(f"{index_path}_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

def save_processed_data(chunks, output_file="nq_trn_processed_data.json"):
    processed_data = []
    for chunk in chunks:
        chunk_dict = chunk.copy()
        chunk_dict['embedding'] = chunk_dict['embedding'].tolist()
        processed_data.append(chunk_dict)
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)

if __name__ == "__main__":
    try:
        nq_file_path = "src/data/biencoder-nq-train.json"
        
        print("Starting process...")
        
        nq_data = load_nq_dataset(nq_file_path)
        processed_data = process_nq_data(nq_data)
        
        chunks = create_chunks_from_nq(processed_data, batch_size=32)
        
        build_and_save_faiss_index(chunks)
        
        save_processed_data(chunks, "src/data/nq_trn_processed_data.json")
        
        print("\nProcess completed. FAISS index and JSON file saved.")
    finally:
        model_manager.cleanup()