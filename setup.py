# setup.py
from setuptools import setup, find_packages

setup(
    name='semantic_cache',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'redis',
        'faiss-cpu',
        'numpy',
        'sentence-transformers',
        'pytest',
        'fakeredis'
    ],
    author='Subhagato Adak',
    author_email='subhagatoadak.india@gmail.com',
    description='A production-grade semantic caching solution for generative AI tools',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
