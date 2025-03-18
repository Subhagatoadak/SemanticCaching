from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # Your current model
embedding = model.encode("test query")

print(f"Embedding shape: {embedding.shape}")  # Check the output
