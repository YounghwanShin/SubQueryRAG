from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
import torch
import json
from resource_manager import ModelManager, cuda_memory_manager
import numpy as np

MAX_CONTEXT_LENGTH = 5000
RELEVANCE_THRESHOLD = 0.35

model_manager = ModelManager()

def get_query_embedding(text: str) -> np.ndarray:
    inputs = model_manager.question_tokenizer(
        text, 
        max_length=256, 
        padding=True, 
        truncation=True,
        return_tensors="pt"
    ).to(model_manager.device)
    
    with cuda_memory_manager():
        with torch.no_grad():
            embeddings = model_manager.question_encoder(**inputs).pooler_output
        return embeddings[0].cpu().numpy()

"""
def load_processed_data(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for item in data:
        item['embedding'] = np.array(item['embedding'])
    
    return data
"""