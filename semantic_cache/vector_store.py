# semantic_cache/vector_store.py
import numpy as np
import faiss
from typing import List, Tuple
import logging
from semantic_cache import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dim: int = config.VECTOR_DIM):
        self.dim = dim
        # Create a flat index and wrap it with IndexIDMap to support deletion.
        self.base_index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIDMap(self.base_index)
        # Maps to keep track of keys and their corresponding integer IDs.
        self.key_to_id = {}
        self.id_to_key = {}
        self.next_id = 0  # Unique integer ID generator.
        logger.info("Initialized FAISS vector store with IndexIDMap wrapping flat L2 index.")

    def add(self, key: str, vector: np.ndarray):
        """
        Add a new vector to the FAISS index with an associated key.
        """
        try:
            vector = vector.astype('float32').reshape(1, self.dim)
            # Assign a unique integer ID for the vector.
            cur_id = self.next_id
            self.next_id += 1
            self.key_to_id[key] = cur_id
            self.id_to_key[cur_id] = key
            # Add the vector along with its integer ID.
            self.index.add_with_ids(vector, np.array([cur_id], dtype=np.int64))
            logger.debug(f"Added vector for key {key} with id {cur_id} to FAISS index.")
        except Exception as e:
            logger.exception(f"Error adding vector for key {key}: {e}")

    def search(self, vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Search for the most similar vector(s) and return a list of (key, distance) tuples.
        """
        try:
            vector = vector.astype('float32').reshape(1, self.dim)
            distances, indices = self.index.search(vector, top_k)
            results = []
            for dist, id in zip(distances[0], indices[0]):
                if id in self.id_to_key:
                    key = self.id_to_key[id]
                    results.append((key, dist))
            logger.debug(f"FAISS search results: {results}")
            return results
        except Exception as e:
            logger.exception(f"Error during FAISS search: {e}")
            return []

    def delete(self, key: str):
        """
        Remove the vector associated with the given key from the FAISS index.
        """
        try:
            if key not in self.key_to_id:
                logger.warning(f"Key {key} not found in vector store.")
                return
            id_to_remove = self.key_to_id[key]
            # Remove the vector using its unique integer ID.
            self.index.remove_ids(np.array([id_to_remove], dtype=np.int64))
            # Remove the key mappings.
            del self.key_to_id[key]
            del self.id_to_key[id_to_remove]
            logger.info(f"Deleted vector for key {key} with id {id_to_remove} from FAISS index.")
        except Exception as e:
            logger.exception(f"Error deleting vector for key {key}: {e}")
