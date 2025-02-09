#!/bin/bash
conda env create -f environment.yml
echo "Conda environment 'se_new' with Python 3.10 created successfully."

conda activate se_new

ollama pull llama3.2:3b # this will take around 2gb best model that we can run locally
ollama pull nomic-embed-text # this will take 1gb to download the model 

# pip install Flask omegaconf langchain langchain-community faiss-cpu tiktoken langchain-ollama