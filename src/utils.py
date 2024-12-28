import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

MAX_CONTEXT_LENGTH = 5000

RELEVANCE_THRESHOLD = 0.35

model = SentenceTransformer('all-mpnet-base-v2')

def get_embedding(text: str) -> List[float]:
    return model.encode(text).tolist()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))