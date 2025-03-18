#!/bin/bash
# Script to create the folder structure for the semantic_cache package

# Create directories
echo "Creating directories..."
mkdir -p semantic_cache tests

# Create files inside semantic_cache folder
echo "Creating files in semantic_cache directory..."
touch semantic_cache/__init__.py
touch semantic_cache/cache_manager.py
touch semantic_cache/config.py
touch semantic_cache/embedding.py
touch semantic_cache/persistent_cache.py
touch semantic_cache/session_cache.py
touch semantic_cache/utils.py
touch semantic_cache/vector_store.py

# Create files inside tests folder
echo "Creating files in tests directory..."
touch tests/test_cache_manager.py
touch tests/test_embedding.py
touch tests/test_persistent_cache.py
touch tests/test_session_cache.py
touch tests/test_vector_store.py

# Create root-level files
echo "Creating root-level files..."
touch setup.py

echo "Folder structure created successfully."
