# A Solution to the CASML Gen AI Competition Challenge

A document question-answering system that implements hybrid retrieval and reranking strategies for answering questions from a  PDF document.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines:
- **Multiple Chunking Strategies**: Page-based, fixed-size, and recursive chunking
- **Hybrid Retrieval**: Semantic embeddings + Keyword-based search (BM25)
- **Reranking**: Cross-encoder models with hypothetical document embeddings (HyDE)
- **LLM-based Answer Generation**: Context-aware answer generation

The system is designed to answer questions from a psychology textbook by retrieving relevant passages and generating accurate, context-consistent answers.

## Features

### Chunking Strategies

1. **Page Chunking** (`page_chunking`): Each page becomes a single chunk
2. **Fixed Chunking** (`fixed_chunking`): Fixed-size chunks with overlap (default: 1000 chars, 100 overlap)
3. **Recursive Chunking** (`recursive_chunking`): Recursive character text splitting preserving sentence boundaries

### Retrieval System

- **Semantic Vector Store**: Uses a transformer embeddings stored in ChromaDB
- **Keyword Vector Store**: BM25-based keyword search using `rank_bm25`
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) to combine results from both retrieval methods

### Reranking

- **Hypothetical Document Embeddings (HyDE)**: Generates hypothetical answers using LLM to enhance query understanding
- **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for fine-grained relevance scoring
- **Score Combination**: Combines RRF scores with cross-encoder scores (50/50 weighted)

### Answer Generation

- Context-first prompting strategy
- Instructions for structured, factual, and detailed answers
- Single-answer enforcement to avoid repetition
- Answer cleaning (removes LaTeX, markdown, excessive whitespace)

## Installation

### Create the conda environment
```bash
conda env create -f environment.yml
conda activate casml
```

### Download the Data

You need to download the `book.pdf` (the psychology textbook) and `queries.json` (the test queries) from the competition data page:

https://www.kaggle.com/competitions/casml-generative-ai-hackathon/data

After downloading, copy both `book.pdf` and `queries.json` into the data folder of this repository. This ensures all scripts and notebooks can correctly locate the data files.


## Usage

### Running Experiments

1. Configure your experiment in `exps/` directory (e.g., `fixed_chunking.yml`)

2. Open and run `langchain_solution.ipynb`:
   - The notebook loads the PDF and metadata
   - Creates vector stores (semantic and keyword-based)
   - Runs retrieval and reranking
   - Generates answers for all queries
   - Saves results to CSV

### Experiment Configuration

Example configuration (`exps/fixed_chunking.yml`):
```yaml
exp_name: "fixed_chunking"
semantic_vs_path: "./data/semantic_vs_fc"
keyword_vs_fname: "./data/keyword_vs_fc.pkl"
chunking: "fixed_chunking"
top_k: 16
out_fname: "top16_fixed_chunking.csv"
```


## Results

Our best model achieved a public score on par with the Top 20% best Kaggle competition submissions.



