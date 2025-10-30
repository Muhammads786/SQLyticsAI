from langchain.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb, os, pathlib, requests

# Simple web search tool (swap to Serper/Bing/etc.)
def web_search(query: str, k: int = 5) -> list[dict]:
    # Placeholder: implement your chosen search API here
    # Return [{"title":..., "url":..., "snippet":...}, ...]
    return []

def fetch_url_text(url: str) -> str:
    try:
        return requests.get(url, timeout=15).text[:200000]
    except Exception:
        return ""

def load_pdf(path: str) -> str:
    chunks = []
    for p in PyPDFLoader(path).load():
        chunks.append(p.page_content)
    return "\n".join(chunks)

def chunk(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return [c for c in splitter.split_text(text)]
