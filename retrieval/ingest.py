"""
retrieval/ingest.py
PDF 파싱 → 청킹 → 임베딩(multilingual-e5-large) → FAISS 인덱스 저장

교재 Agentic RAG 패턴 적용:
- 문서별 메타데이터(source, page, doc_type) 부착
- 100페이지 제한 준수 (config.py 의 PDF_PAGE_RANGES 기준)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, FAISS_DIR,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    PDF_PAGE_RANGES,
)


# ─── PDF Parsing ──────────────────────────────────────────────────────────────

def parse_pdf(
    pdf_path: str,
    page_range: Optional[Tuple[int, int]] = None,
    doc_type: str = "unknown",
) -> List[Document]:
    """
    PDF를 파싱하여 LangChain Document 리스트로 반환.
    page_range: (start_page, end_page) 1-indexed, inclusive
    """
    docs: List[Document] = []
    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)

    start = (page_range[0] - 1) if page_range else 0
    end = min(page_range[1], total_pages) if page_range else total_pages

    print(f"  Parsing: {Path(pdf_path).name}  (pages {start+1}~{end} / {total_pages})")

    for page_num in range(start, end):
        page = pdf[page_num]
        text = page.get_text("text").strip()
        if not text or len(text) < 30:
            continue  # skip empty / image-only pages

        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(pdf_path),
                "filename": Path(pdf_path).name,
                "page": page_num + 1,          # 1-indexed
                "doc_type": doc_type,           # "lges" | "catl" | "market"
            }
        ))

    pdf.close()
    return docs


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """RecursiveCharacterTextSplitter 기반 청킹 (메타데이터 보존)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Chunking: {len(docs)} pages → {len(chunks)} chunks")
    return chunks


# ─── Embeddings ───────────────────────────────────────────────────────────────

def load_embeddings() -> HuggingFaceEmbeddings:
    """
    intfloat/multilingual-e5-large 로드.
    한국어/영어/중국어 멀티링컬 지원 (오픈소스, 무료).
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
    )
    return embeddings


# ─── FAISS Index ──────────────────────────────────────────────────────────────

def build_faiss_index(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_name: str,
) -> FAISS:
    """청크 → FAISS 인덱스 생성 및 디스크 저장."""
    save_path = FAISS_DIR / index_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"  Building FAISS index '{index_name}' ({len(chunks)} chunks)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(save_path))
    print(f"  Saved: {save_path}")
    return vectorstore


def load_faiss_index(
    index_name: str,
    embeddings: HuggingFaceEmbeddings,
) -> Optional[FAISS]:
    """저장된 FAISS 인덱스 로드. 없으면 None 반환."""
    save_path = FAISS_DIR / index_name
    if (save_path / "index.faiss").exists():
        print(f"  Loading existing FAISS index: {index_name}")
        return FAISS.load_local(
            str(save_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


# ─── Main Ingestion Pipeline ──────────────────────────────────────────────────

def ingest_doc_type(
    doc_type: str,
    embeddings: HuggingFaceEmbeddings,
    force_rebuild: bool = False,
) -> FAISS:
    """
    특정 doc_type(lges | catl | market)의 PDF를 수집·청킹·인덱싱.
    force_rebuild=False 이면 기존 인덱스 재사용.
    """
    cfg = PDF_PAGE_RANGES[doc_type]
    data_path: Path = cfg["path"]
    page_range: Tuple[int, int] = cfg["page_range"]
    index_name = f"faiss_{doc_type}"

    # 기존 인덱스 존재하면 재사용
    if not force_rebuild:
        vs = load_faiss_index(index_name, embeddings)
        if vs is not None:
            return vs

    print(f"\n[Ingest] {doc_type.upper()} — {cfg['description']}")

    # PDF 파일 수집
    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_path}\n"
            f"  → {doc_type} 폴더에 PDF 파일을 복사해주세요."
        )

    all_docs: List[Document] = []
    for pdf_path in pdf_files:
        docs = parse_pdf(str(pdf_path), page_range=page_range, doc_type=doc_type)
        all_docs.extend(docs)

    chunks = chunk_documents(all_docs)
    vectorstore = build_faiss_index(chunks, embeddings, index_name)
    return vectorstore


def ingest_all(force_rebuild: bool = False) -> Dict[str, FAISS]:
    """
    lges / catl / market 세 가지 인덱스를 모두 구축.
    Returns: {"lges": FAISS, "catl": FAISS, "market": FAISS}
    """
    print("=" * 60)
    print("Battery Strategy RAG — Document Ingestion Pipeline")
    print("=" * 60)

    embeddings = load_embeddings()
    vectorstores: Dict[str, FAISS] = {}

    for doc_type in ["lges", "catl", "market"]:
        vectorstores[doc_type] = ingest_doc_type(doc_type, embeddings, force_rebuild)

    print("\n✅ All indexes built successfully.")
    return vectorstores


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS indexes")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild all indexes")
    parser.add_argument("--doc-type", choices=["lges", "catl", "market", "all"], default="all")
    args = parser.parse_args()

    embeddings = load_embeddings()

    if args.doc_type == "all":
        ingest_all(force_rebuild=args.rebuild)
    else:
        ingest_doc_type(args.doc_type, embeddings, force_rebuild=args.rebuild)
