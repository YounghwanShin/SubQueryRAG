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

def create_chunks_from_nq(processed_data):
    chunks = []
    for idx, item in enumerate(tqdm(processed_data, desc="Creating chunks and embeddings")):
        chunk_dict = item.copy()
        text = text_formatter(item['text'])
        
        inputs = model_manager.context_tokenizer(
            text, 
            max_length=256, 
            padding=True, 
            truncation=True,
            return_tensors="pt"
        ).to(model_manager.device)
        
        with cuda_memory_manager():
            with torch.no_grad():
                embedding = model_manager.context_encoder(**inputs).pooler_output[0]
                chunk_dict["embedding"] = embedding.cpu().numpy()
        
        chunks.append(chunk_dict)
    
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

def save_processed_data(chunks, output_file="nq_processed_data.json"):
    processed_data = []
    for chunk in chunks:
        chunk_dict = chunk.copy()
        chunk_dict['embedding'] = chunk_dict['embedding'].tolist()
        processed_data.append(chunk_dict)
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)

if __name__ == "__main__":
    try:
        nq_file_path = "src/data/nq_test.json"
        
        print("Starting process...")
        
        nq_data = load_nq_dataset(nq_file_path)
        processed_data = process_nq_data(nq_data)
        chunks = create_chunks_from_nq(processed_data)
        
        build_and_save_faiss_index(chunks)
        
        save_processed_data(chunks, "src/data/nq_processed_data.json")
        
        print("\nProcess completed. FAISS index and JSON file saved.")
    finally:
        model_manager.cleanup()