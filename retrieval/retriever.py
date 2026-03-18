"""
retrieval/retriever.py
FAISS 벡터스토어 기반 문서 검색 모듈.

교재 Agentic RAG 다이어그램 구현:
  Query Transformation → Retrieve → Grade Documents
  → (NO) Query Rewrite → Retrieve  (max_retry 횟수만큼 반복)
  → (YES) context 반환
"""

import json
from typing import List, Optional, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    FAISS_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD, OPENAI_API_KEY,
    LLM_MODEL, LLM_TEMPERATURE,
)


# ─── Embedding Loader ─────────────────────────────────────────────────────────

_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True, "batch_size": EMBEDDING_BATCH_SIZE},
        )
    return _embeddings


# ─── Index Loader ─────────────────────────────────────────────────────────────

_indexes: Dict[str, FAISS] = {}

def get_index(doc_type: str) -> FAISS:
    """doc_type 에 해당하는 FAISS 인덱스 로드 (캐싱)."""
    if doc_type not in _indexes:
        index_path = FAISS_DIR / f"faiss_{doc_type}"
        if not (index_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                f"  → 먼저 'python retrieval/ingest.py' 를 실행해 주세요."
            )
        _indexes[doc_type] = FAISS.load_local(
            str(index_path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _indexes[doc_type]


# ─── Document Grader ──────────────────────────────────────────────────────────

_grader_llm: Optional[ChatOpenAI] = None

def get_grader_llm() -> ChatOpenAI:
    global _grader_llm
    if _grader_llm is None:
        _grader_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )
    return _grader_llm


BATCH_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document relevance grader. "
     "Given a question and a numbered list of documents, "
     "return a JSON array of YES/NO verdicts — one per document — indicating relevance. "
     "Return ONLY the JSON array, nothing else. Example: [\"YES\", \"NO\", \"YES\"]"),
    ("human",
     "Question: {question}\n\n"
     "Documents:\n{documents}\n\n"
     "Return JSON array of YES/NO (one per document):"),
])


def batch_grade_documents(question: str, docs: List[Document]) -> List[Document]:
    """
    k개 문서를 LLM 1회 호출로 일괄 관련성 평가 (성능 최적화).

    기존 grade_document() 대비 k배 빠름:
      k=4 문서 → LLM 4회 호출 → LLM 1회 호출
    """
    if not docs:
        return []
    doc_texts = "\n\n".join(
        [f"[{i+1}] {doc.page_content[:400]}" for i, doc in enumerate(docs)]
    )
    chain = BATCH_GRADE_PROMPT | get_grader_llm()
    result = chain.invoke({"question": question, "documents": doc_texts})
    try:
        text = result.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        verdicts = json.loads(text)
        return [doc for doc, v in zip(docs, verdicts) if str(v).upper().startswith("YES")]
    except (json.JSONDecodeError, TypeError, IndexError):
        return docs  # fallback: 모든 문서 반환


# ─── Query Rewriter ───────────────────────────────────────────────────────────

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimization expert for battery/energy industry documents. "
     "Rewrite the given query to improve document retrieval. "
     "Make it more specific and use relevant technical keywords. "
     "Return ONLY the rewritten query, nothing else."),
    ("human",
     "Original query: {query}\n"
     "Previous retrieval returned irrelevant results.\n"
     "Rewritten query:"),
])

def rewrite_query(query: str) -> str:
    """검색 결과 미흡 시 쿼리 재작성 (Query Rewrite 단계)."""
    chain = REWRITE_PROMPT | get_grader_llm()
    result = chain.invoke({"query": query})
    return result.content.strip()


# ─── Core Retrieval Function ──────────────────────────────────────────────────

def retrieve_with_grading(
    query: str,
    doc_type: str,
    k: int = RETRIEVAL_K,
    max_retry: int = 5,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> List[Document]:
    """
    교재 다이어그램 구현: 검색 → 문서 평가 → 재검색 루프

    Diagram (Textbook p.2):
      Retrieve → Grade Documents → YES: return / NO: Query Rewrite → Retrieve (반복)

    Args:
        query: 검색 쿼리
        doc_type: "lges" | "catl" | "market"
        k: 검색할 문서 수
        max_retry: 최대 재검색 횟수 (교재 max_retry: 5)
        score_threshold: 최소 유사도 점수
    Returns:
        관련성 있다고 판단된 Document 리스트
    """
    index = get_index(doc_type)
    current_query = query

    for attempt in range(max_retry + 1):
        # 1. Retrieve
        results_with_scores = index.similarity_search_with_score(current_query, k=k)

        # 필터링: 점수 기반 (FAISS는 L2 distance, 낮을수록 유사)
        candidates = [
            doc for doc, score in results_with_scores
            if score < (1.0 - score_threshold)  # cosine sim → L2 distance 변환 근사
        ]

        if not candidates:
            candidates = [doc for doc, _ in results_with_scores]  # fallback

        # 2. Grade Documents (배치 처리 — k회 → 1회 LLM 호출)
        relevant_docs = batch_grade_documents(current_query, candidates)

        if relevant_docs:
            # YES: 관련 문서 반환
            return relevant_docs

        if attempt < max_retry:
            # NO: Query Rewrite → 재검색
            current_query = rewrite_query(current_query)

    # max_retry 소진 → 가장 높은 유사도 문서 반환 (fallback)
    return [doc for doc, _ in results_with_scores[:k]]


def retrieve_context(
    query: str,
    doc_type: str,
    k: int = RETRIEVAL_K,
    max_retry: int = 5,
) -> str:
    """검색 결과를 하나의 컨텍스트 문자열로 합쳐 반환."""
    docs = retrieve_with_grading(query, doc_type, k=k, max_retry=max_retry)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[Source {i}: {meta.get('filename','?')}, p.{meta.get('page','?')}]"
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)
