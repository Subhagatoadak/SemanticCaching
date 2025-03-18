import multiprocessing
import numpy as np
import faiss
import logging
import gc
from semantic_cache import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """ Initialize FAISS with correct dimensions and support for ID-based indexing. """
        self.dim = config.VECTOR_DIM
        self.base_index = faiss.IndexFlatIP(self.dim)  # Use Inner Product (IP) instead of L2
        self.index = faiss.IndexIDMap(self.base_index)
        self.key_to_id = {}
        self.id_to_key = {}
        self.next_id = 0
        logger.info(f"Initialized FAISS vector store with IndexIDMap, dimension: {self.dim}")

    def _run_faiss_task(self, method_name, *args):
        """ Run FAISS operations in a separate process to prevent segmentation faults. """
        process = multiprocessing.Process(target=getattr(self, method_name), args=args)
        process.start()
        process.join()

    def add_vector(self, key, vector):
        """ Add a new vector with a unique ID to FAISS. """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: Expected {self.dim}, got {vector.shape[0]}")

        vector = vector.reshape(1, -1).astype("float32")
        cur_id = self.next_id
        self.next_id += 1

        try:
            self.index.add_with_ids(vector, np.array([cur_id], dtype=np.int64))
            self.key_to_id[key] = cur_id
            self.id_to_key[cur_id] = key
            logger.info(f"Added vector for key {key} with ID {cur_id}")
        except Exception as e:
            logger.error(f"Error adding vector for key {key}: {e}")

    def add(self, key, vector):
        """ Run FAISS add operation inside a subprocess. """
        self._run_faiss_task("add_vector", key, vector)

    def search_vectors(self, vector, top_k, results):
        """ Search for the most similar vector(s) in FAISS. """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: Expected {self.dim}, got {vector.shape[0]}")

        vector = vector.reshape(1, -1).astype("float32")

        try:
            distances, indices = self.index.search(vector, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.id_to_key:
                    results.append((self.id_to_key[idx], dist))
            logger.info(f"FAISS search results: {list(results)}")
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")

    def search(self, vector, top_k=1):
        """ Run FAISS search inside a subprocess. """
        results = multiprocessing.Manager().list()
        process = multiprocessing.Process(target=self.search_vectors, args=(vector, top_k, results))
        process.start()
        process.join()
        return list(results)

    def delete_vector(self, key):
        """ Remove a vector from FAISS by key. """
        if key not in self.key_to_id:
            logger.warning(f"Key {key} not found in FAISS index.")
            return

        id_to_remove = self.key_to_id[key]

        try:
            self.index.remove_ids(np.array([id_to_remove], dtype=np.int64))
            del self.key_to_id[key]
            del self.id_to_key[id_to_remove]
            gc.collect()
            logger.info(f"Deleted vector for key {key} (ID {id_to_remove}) from FAISS index.")
        except Exception as e:
            logger.error(f"Error deleting vector for key {key}: {e}")

    def delete(self, key):
        """ Run FAISS delete inside a subprocess. """
        self._run_faiss_task("delete_vector", key)

    def reset_index_process(self):
        """ Internal method to reset FAISS index inside a subprocess. """
        self.base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.base_index)
        self.key_to_id.clear()
        self.id_to_key.clear()
        self.next_id = 0
        gc.collect()
        logger.info("FAISS index has been reset.")

    def reset_index(self):
        """ Run FAISS reset inside a subprocess. """
        self._run_faiss_task("reset_index_process")
