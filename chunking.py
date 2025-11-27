# Load the page metadatas

import pandas as pd
import os
import csv

from typing import List

from typing import TypedDict
#from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re
from bisect import bisect_right

def clean_page_text(page_text: str) -> str:
    # Remove newlines and excessive spaces
    cleaned_text = ' '.join(page_text.split())
    # Remove URLs using regular expression
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
    cleaned_text = re.sub(r'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
    return cleaned_text


def page_chunking(page_to_section, pages, **kwargs):
    docs = []
    for i in range(len(pages)):
        if (i in page_to_section.keys()) and (page_to_section[i]["subsection"] is not None):
            metadata = {
                "id": i,
                "page_num": i,
                "sections": f'{page_to_section[i]["section"]}/{page_to_section[i]["subsection"]}'
            }
            doc = Document(page_content=clean_page_text(pages[i]), metadata=metadata)
            docs.append(doc)

    return docs

def _concatenate_pages(pages: List[str]) -> str:
    texts = []
    offsets = []
    pos = 0
    for i, page in enumerate(pages):
        text = clean_page_text(page)
        offsets.append(pos)
        texts.append(text)
        # add 1 for the separator we'll use when joining pages
        pos += len(text) + 1

    if not texts:
        return None, None

    concatenated = " ".join(texts)
    return concatenated, offsets

def fixed_chunking(page_to_section, pages, chunk_size = 1024, overlap_prop = 0.1, start_page=0):
    """
    Load a PDF from filepath, concatenate all pages, split into chunks of chunk_size,
    and attach metadata.page_num as the page index where each chunk starts.
    """
    concatenated, offsets = _concatenate_pages(pages)

    docs = []
    total_len = len(concatenated)
    overlap = int(chunk_size * overlap_prop)
    
    for start in range(0, total_len, chunk_size):
        chunk = concatenated[max(0, start - overlap):start + chunk_size]
        page_idx = bisect_right(offsets, start) - 1
        
        if page_idx < 0:
            page_idx = 0

        page_idx += start_page
        
        if page_idx in page_to_section.keys() and page_to_section[page_idx]["subsection"] is None:
            # Adjust page index based on start_page
            metadata = {
                    "id": start,
                    "page_num": page_idx,
                    "sections": f'{page_to_section[page_idx]["section"]}/{page_to_section[page_idx]["subsection"]}'
            }
            
            docs.append(Document(page_content=chunk, metadata=metadata))

    return docs


def recursive_chunking(page_to_section, pages, chunk_size=1024, overlap_prop=0.1, start_page=0):
    text, offsets = _concatenate_pages(pages)

    overlap = int(chunk_size * overlap_prop)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)

    start_idx = 0
    docs = []
    for chunk in chunks:
        # Determine which page this chunk starts on
        page_idx = bisect_right(offsets, start_idx) - 1

        if page_idx < 0:
            page_idx = 0
        page_idx += start_page

        if page_idx in page_to_section.keys() and page_to_section[page_idx]["subsection"] is not None:
            metadata = {
                "id": start_idx,
                "page_num": page_idx,
                "sections": f'{page_to_section[page_idx]["section"]}/{page_to_section[page_idx]["subsection"]}'
            }

            doc = Document(page_content=chunk, metadata=metadata)
            docs.append(doc)
        
        start_idx += len(chunk) + 1  # +1 for the space added during concatenation
    
    return docs


def get_chunking_fn(chunking_method):
    if chunking_method == "page_chunking":
        return page_chunking
    elif chunking_method == "fixed_chunking":
        return fixed_chunking
    elif chunking_method == "recursive_chunking":
        return recursive_chunking
    else:
        raise ValueError(f"Unknown chunking method: {chunking_method}")