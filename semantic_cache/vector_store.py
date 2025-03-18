import multiprocessing
import numpy as np
import faiss
import logging
import gc
from typing import Any, List, Tuple
from multiprocessing.managers import ListProxy  # Correct import for type annotation
from semantic_cache import config

logger = logging.getLogger(__name__)

class VectorStore:
    """
    VectorStore wraps FAISS to support adding, searching, deleting, and resetting vectors.
    Operations are executed in subprocesses to isolate the FAISS C++ backend and avoid segmentation faults.
    """
    def __init__(self) -> None:
        self.dim: int = config.VECTOR_DIM
        # Use Inner Product (IP) for stability
        self.base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.base_index)
        self.key_to_id: dict[str, int] = {}
        self.id_to_key: dict[int, str] = {}
        self.next_id: int = 0
        logger.info(f"Initialized FAISS vector store with IndexIDMap, dimension: {self.dim}")

    def _run_faiss_task(self, method_name: str, *args: Any, timeout: int = 10) -> None:
        """
        Run a FAISS operation in a subprocess with a timeout.
        """
        process = multiprocessing.Process(target=getattr(self, method_name), args=args)
        process.start()
        process.join(timeout)
        if process.is_alive():
            logger.error(f"FAISS task '{method_name}' timed out. Terminating process.")
            process.terminate()
            process.join()
        if process.exitcode not in (0, None):
            logger.error(f"FAISS task '{method_name}' failed with exit code {process.exitcode}.")

    def add_vector(self, key: str, vector: np.ndarray) -> None:
        """
        Add a new vector with a unique ID to FAISS.
        """
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

    def add(self, key: str, vector: np.ndarray) -> None:
        """
        Run the add_vector operation in a subprocess.
        """
        self._run_faiss_task("add_vector", key, vector)

    def search_vectors(self, vector: np.ndarray, top_k: int, results: ListProxy) -> None:
        """
        Search for the most similar vectors in FAISS and append results to the shared list.
        
        Args:
            vector (np.ndarray): The query vector.
            top_k (int): The number of top results to retrieve.
            results (ListProxy): A shared list to store (key, distance) tuples.
        """
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

    def search(self, vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Run the FAISS search operation in a subprocess and return the results.
        """
        manager = multiprocessing.Manager()
        results: ListProxy = manager.list()
        process = multiprocessing.Process(target=self.search_vectors, args=(vector, top_k, results))
        process.start()
        process.join(10)
        if process.is_alive():
            logger.error("FAISS search timed out. Terminating process.")
            process.terminate()
            process.join()
        return list(results)

    def delete_vector(self, key: str) -> None:
        """
        Delete the vector corresponding to the provided key from FAISS.
        """
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

    def delete(self, key: str) -> None:
        """
        Run the delete_vector operation in a subprocess.
        """
        self._run_faiss_task("delete_vector", key)

    def reset_index_process(self) -> None:
        """
        Internal method to reset the FAISS index; to be run in a subprocess.
        """
        self.base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.base_index)
        self.key_to_id.clear()
        self.id_to_key.clear()
        self.next_id = 0
        gc.collect()
        logger.info("FAISS index has been reset.")

    def reset_index(self) -> None:
        """
        Run the reset_index_process operation in a subprocess.
        """
        self._run_faiss_task("reset_index_process")
