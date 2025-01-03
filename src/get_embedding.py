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
    """Process a batch of items and generate embeddings"""
    batch_texts = [text_formatter(item['text']) for item in batch_items]
    
    # Tokenize all texts in the batch
    inputs = model_manager.context_tokenizer(
        batch_texts,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model_manager.device)
    
    # Generate embeddings for the entire batch
    with cuda_memory_manager():
        with torch.no_grad():
            embeddings = model_manager.context_encoder(**inputs).pooler_output
            embeddings = embeddings.cpu().numpy()
    
    # Add embeddings to the items
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
        
        # Using a batch size of 32, but this can be adjusted based on available GPU memory
        chunks = create_chunks_from_nq(processed_data, batch_size=32)
        
        build_and_save_faiss_index(chunks)
        
        save_processed_data(chunks, "src/data/nq_trn_processed_data.json")
        
        print("\nProcess completed. FAISS index and JSON file saved.")
    finally:
        model_manager.cleanup()import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import re

nlp = English()
nlp.add_pipe("sentencizer")

def text_formatter(text):
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_file(file_path):
    pages_and_texts = []
    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, desc="Reading file")):
            text = text_formatter(line)
            if text:
                pages_and_texts.append({
                    "page_number": idx,
                    "page_char_count": len(text),
                    "page_word_count": len(text.split()),
                    "page_setence_count_raw": len(text.split(". ")),
                    "page_token_count": len(text) / 4,
                    "text": text
                })
    return pages_and_texts

def analyze_token_lengths(pages_and_texts):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sentence_token_lengths = []
    
    for item in tqdm(pages_and_texts, desc="Analyzing tokens"):
        text = item["text"]
        sentences = [str(sent) for sent in nlp(text).sents]
        for sentence in sentences:
            tokens = tokenizer(sentence, truncation=False)
            sentence_token_lengths.append(len(tokens['input_ids']))
    
    lengths_array = np.array(sentence_token_lengths)
    print(f"\nToken length statistics:")
    print(f"Mean: {lengths_array.mean():.2f}")
    print(f"Median: {np.median(lengths_array):.2f}")
    print(f"95th percentile: {np.percentile(lengths_array, 95):.2f}")
    print(f"Max: {lengths_array.max():.2f}")
    
    return lengths_array

def process_pages(pages_and_texts, slice_size=10):
    for item in tqdm(pages_and_texts, desc="Processing pages"):
        item["sentences"] = [str(sentence) for sentence in nlp(item["text"]).sents]
        item["page_sentence_count_spacy"] = len(item["sentences"])
        item["sentence_chunks"] = [item["sentences"][i:i + slice_size] for i in range(0, len(item["sentences"]), slice_size)]
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

def create_chunks(pages_and_texts):
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Creating chunks"):
        for sentence_chunk in item["sentence_chunks"]:
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', "".join(sentence_chunk).replace("  ", " ").strip())
            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": joined_sentence_chunk,
                "chunk_char_count": len(joined_sentence_chunk),
                "chunk_word_count": len(joined_sentence_chunk.split()),
                "chunk_token_count": len(joined_sentence_chunk) / 4
            }
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def create_embeddings(pages_and_chunks, model_name="all-mpnet-base-v2", device="cpu"):
    embedding_model = SentenceTransformer(model_name, device=device).to(device)

    for item in tqdm(pages_and_chunks, desc="Creating embeddings"):
        embedding = embedding_model.encode(item["sentence_chunk"])
        item["embedding"] = embedding
    
    return pages_and_chunks

if __name__ == "__main__":    
    file_path = "nq_raw.txt"
    
    print("Starting process...")
    pages_and_texts = open_and_read_file(file_path)
    
    token_lengths = analyze_token_lengths(pages_and_texts)
    
    avg_tokens_per_sentence = np.mean(token_lengths)
    recommended_slice_size = int(384 / avg_tokens_per_sentence)
    print(f"\nRecommended slice_size: {recommended_slice_size}")
    
    pages_and_texts = process_pages(pages_and_texts, slice_size=recommended_slice_size)
    pages_and_chunks = create_chunks(pages_and_texts)
    pages_and_chunks = create_embeddings(pages_and_chunks, device="cpu")

    df = pd.DataFrame(pages_and_chunks)
    df.to_csv("text_chunks_and_embeddings_df.csv", index=False)
    
    print("\nProcess completed. CSV file saved.")