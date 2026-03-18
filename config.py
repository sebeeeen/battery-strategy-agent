"""
Global configuration for Battery Strategy Analysis Multi-Agent System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FAISS_DIR = BASE_DIR / "faiss_index"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LLM Settings
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
LLM_TEMPERATURE = 0.0
LLM_TEMPERATURE_REPORT = 0.3  # slightly creative for report writing

# Embedding Settings
# Open-source multilingual embedding (Korean/English/Chinese support)
# BAAI/bge-m3: 100+ 언어 지원, 한국어 성능 우수, 무료 오픈소스
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = "cpu"           # change to "cuda" if GPU available
EMBEDDING_BATCH_SIZE = 32

# PDF Page Selection (100-page total limit)
# Pages are 0-indexed internally, but 1-indexed in config for readability
PDF_PAGE_RANGES = {
    "lges": {
        "path": DATA_DIR / "lges",
        "page_range": (1, 40),      # LGES ESG 2024: pp.1-40 (strategy/overview)
        "description": "LG Energy Solution 2024 ESG Report",
    },
    "catl": {
        "path": DATA_DIR / "catl",
        "page_range": (1, 40),      # CATL 2023 ESG: pp.1-40
        "description": "CATL 2023 ESG Report",
    },
    "market": {
        "path": DATA_DIR / "market",
        "page_range": (11, 30),     # IEA EV Outlook: Executive Summary + EV Trends
        "description": "IEA Global EV Outlook 2024",
    },
}

# ─── Chunking Settings ────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # overlap between chunks

# ─── Retrieval Settings ───────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("TOP_K_TECH", "4"))          # top-k documents per retrieval call
RETRIEVAL_K_MARKET = int(os.getenv("TOP_K_MARKET", "4")) # top-k for market context
RETRIEVAL_SCORE_THRESHOLD = 0.3   # minimum similarity score

# ─── Agentic RAG Settings (from textbook workflow diagrams) ───────────────────
MAX_RETRIEVAL_RETRY = 2           # max query rewrite retries (Search Evaluation)
MAX_REVISION = 2                  # max draft revisions (Self-Reflection)

# ─── Web Search Settings ──────────────────────────────────────────────────────
SEARCH_MAX_RESULTS = 5            # results per query
SEARCH_TOPICS = [
    "LGES ESS strategy",
    "LGES portfolio diversification",
    "CATL ESS strategy",
    "CATL portfolio diversification",
    "global battery market 2024",
]

# ─── Report Settings ──────────────────────────────────────────────────────────
REPORT_LANGUAGE = "Korean"
REPORT_FILENAME = "agent_report.md"
REPORT_PDF_FILENAME = "agent_report.pdf"
