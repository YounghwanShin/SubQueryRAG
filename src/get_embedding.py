import pandas as pd
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