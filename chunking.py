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
                "page_num": i,
                "sections": f'{page_to_section[i]["section"]}/{page_to_section[i]["subsection"]}'
            }
            doc = Document(page_content=clean_page_text(pages[i]), metadata=metadata)
            docs.append(doc)

    return docs

def fixed_chunking(page_to_section, pages, chunk_size=1000, overlap=100):
    docs = []
    for i in range(BOOK_PAGE_OFFSET + 7, BOOK_LAST_PAGE + 1):
        if (i in page_to_section.keys()) and (page_to_section[i]["subsection"] is not None):
            text = clean_page_text(pages[i])
            metadata = {
                "page_num": i,
                "sections": f'{page_to_section[i]["section"]}/{page_to_section[i]["subsection"]}'
            }

            chunks = [text[j:min(j + chunk_size, len(text))] for j in range(0, len(text), chunk_size - overlap)]
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=metadata)
                docs.append(doc)
    return docs


def recursive_chunking(page_to_section, pages, chunk_size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                   chunk_overlap=100)
    docs = []
    for i in range(BOOK_PAGE_OFFSET + 7, BOOK_LAST_PAGE + 1):
        if (i in page_to_section.keys()) and (page_to_section[i]["subsection"] is not None):
            text = clean_page_text(pages[i])
            metadata = {
                "page_num": i,
                "sections": f'{page_to_section[i]["section"]}/{page_to_section[i]["subsection"]}'
            }
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=metadata)
                docs.append(doc)
    return docs

def get_chunking_fn(chunking_method):
    if chunking_method == "page_chunking":
        return page_chunking
    elif chunking_method == "fixed_chunking":
        return fixed_chunking
    elif chunking_method == "recursive_chunking":
        return recursive_chunking