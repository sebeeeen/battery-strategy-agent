"""
agents/lges_rag_agent.py
LG에너지솔루션(LGES) 전용 RAG Agent — 완전 독립 클래스.

[교수님 피드백 반영]
파라미터만 다른 동일 클래스 사용 시 발생하는 데이터 혼합 위험을 방지하기 위해
LGESRagAgent를 RAGAgent와 독립된 별도 클래스로 구현.

분리 근거:
  - 독립된 FAISS 인덱스 참조 (faiss_lges) → CATL 문서와 혼용 불가
  - LGES 전용 시스템 프롬프트 및 분석 관점
  - @tool 기반 retrieve_lges_documents 전용 도구 바인딩
  - 독립된 LLM 인스턴스 → 상태 격리
  - 별도 토픽 목록 (LGES 전략 특화)

교재 Agentic RAG 워크플로우 (완전 구현):
  Query Transformation → Retrieve (LGES FAISS만)
  → Grade Documents → Query Rewrite (max_retry: 5)
  → Draft Generation
  → Self-Reflection (APPROVED / REVISE / RETRIEVE)
    → REVISE: Draft Revision (max_revision: 3)
    → RETRIEVE: 추가 컨텍스트 확보
  → Memory Update (state 반환)
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    MAX_RETRIEVAL_RETRY, MAX_REVISION,
)
from retrieval.retriever import retrieve_context
from prompts.rag import (
    RAG_EXTRACTION_PROMPT,
    RAG_SELF_REFLECTION_PROMPT,
    RAG_REVISION_PROMPT,
)
from agents.tools import retrieve_lges_documents


# ─── LGES 전용 토픽 ────────────────────────────────────────────────────────────

LGES_TOPICS = [
    "포트폴리오 다각화 전략 및 핵심 사업 방향",
    "ESS(에너지저장장치) 사업 전략",
    "주요 경쟁력 및 기술 역량",
]


# ─── LGESRagAgent ─────────────────────────────────────────────────────────────

class LGESRagAgent:
    """
    LG에너지솔루션 전용 Agentic RAG Agent.

    독립 클래스 설계 원칙:
    - LGES FAISS 인덱스만 접근 (faiss_lges 디렉토리)
    - CATL/Market 인덱스와 물리적으로 분리
    - retrieve_lges_documents @tool 바인딩으로 LLM 자율 검색 지원
    - LGES 기업 컨텍스트 특화 프롬프트

    Self-Reflection 판정 기준:
    - APPROVED: 충분한 데이터와 분석이 포함된 고품질 드래프트
    - REVISE: 데이터는 있으나 분석 깊이/구체성 부족
    - RETRIEVE: 핵심 정보 누락으로 추가 문서 검색 필요
    """

    # 클래스 고유 식별자 (데이터 혼합 방지 검증용)
    DOC_TYPE = "lges"
    COMPANY_NAME = "LG에너지솔루션 (LG Energy Solution)"
    INDEX_NAME = "faiss_lges"

    def __init__(self):
        # LLM 인스턴스 — CATLRagAgent와 완전히 분리된 독립 인스턴스
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        # @tool 바인딩 — LGES 전용 검색 도구만 바인딩
        # bind_tools()로 LLM이 필요 시 자율적으로 도구 호출 가능
        self.llm_with_tools = self.llm.bind_tools([retrieve_lges_documents])

        self._log_prefix = "[LGESRagAgent]"

    def _log(self, msg: str):
        print(f"{self._log_prefix} {msg}")

    # ── 핵심 메서드 ─────────────────────────────────────────────────────────────

    def _transform_query(self, topic: str) -> str:
        """
        Query Transformation — LGES 특화 쿼리로 변환.
        회사명 + 토픽 결합으로 검색 정밀도 향상.
        """
        return f"{self.COMPANY_NAME} {topic}"

    def _retrieve(self, query: str, max_retry: int) -> str:
        """
        Retrieve (LGES 인덱스 전용) + Grade Documents.
        retriever.py의 CRAG 루프 활용 — max_retry 횟수만큼 Query Rewrite 반복.
        """
        return retrieve_context(
            query,
            doc_type=self.DOC_TYPE,   # "lges" — CATL 인덱스 접근 불가
            max_retry=max_retry,
        )

    def _extract_draft(self, topic: str, context: str) -> str:
        """Draft Generation — LGES 컨텍스트에서 전략 정보 추출."""
        chain = RAG_EXTRACTION_PROMPT | self.llm
        result = chain.invoke({
            "company": self.COMPANY_NAME,
            "topic": topic,
            "context": context,
        })
        return result.content

    def _self_reflect(self, topic: str, draft: str, context: str) -> dict:
        """
        Self-Reflection — 드래프트 품질 자체 평가.
        반환: {"verdict": "APPROVED"|"REVISE"|"RETRIEVE", "reason": str, ...}
        """
        chain = RAG_SELF_REFLECTION_PROMPT | self.llm
        result = chain.invoke({
            "topic": topic,
            "company": self.COMPANY_NAME,
            "draft": draft,
            "context": context,
        })
        text = result.content.strip()
        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError:
            return {"verdict": "APPROVED", "reason": "JSON parse fallback"}

    def _revise_draft(self, topic: str, draft: str, guidance: str, context: str) -> str:
        """Draft Revision — Self-Reflection 피드백 기반 수정."""
        chain = RAG_REVISION_PROMPT | self.llm
        result = chain.invoke({
            "company": self.COMPANY_NAME,
            "topic": topic,
            "draft": draft,
            "guidance": guidance,
            "context": context,
        })
        return result.content

    # ── 공개 메서드 ─────────────────────────────────────────────────────────────

    def _run_single_topic(self, topic: str, max_retry: int, max_revision: int) -> str:
        """
        단일 토픽 RAG 파이프라인 실행 (병렬 처리 단위).

        Query Transformation → Retrieve → Grade → Draft → Self-Reflection Loop
        """
        query = self._transform_query(topic)
        context = self._retrieve(query, max_retry=max_retry)
        if not context.strip():
            return ""

        draft = self._extract_draft(topic, context)
        revision_count = 0

        while revision_count < max_revision:
            reflection = self._self_reflect(topic, draft, context)
            verdict = reflection.get("verdict", "APPROVED")

            if verdict == "APPROVED":
                break
            elif verdict == "REVISE":
                guidance = reflection.get("revision_guidance", "Improve data specificity")
                draft = self._revise_draft(topic, draft, guidance, context)
                revision_count += 1
            elif verdict == "RETRIEVE":
                missing = reflection.get("missing_info", "")
                new_query = f"{self.COMPANY_NAME} {topic} {missing}"
                extra = self._retrieve(new_query, max_retry=max_retry)
                if extra:
                    context = context + "\n\n---\n\n" + extra
                draft = self._extract_draft(topic, context)
                revision_count += 1
            else:
                break

        return f"## {topic}\n\n{draft}"

    def run(
        self,
        topics: Optional[List[str]] = None,
        max_retry: int = MAX_RETRIEVAL_RETRY,
        max_revision: int = MAX_REVISION,
    ) -> str:
        """
        LGES 전용 Agentic RAG 파이프라인 실행.

        토픽별 병렬 처리 (ThreadPoolExecutor):
          기존: topic-1 → topic-2 → topic-3 (순차)
          변경: topic-1 ┐
                topic-2 ├── 동시 실행 → 합산
                topic-3 ┘
        """
        topics = topics or LGES_TOPICS
        results_map = {}

        with ThreadPoolExecutor(max_workers=len(topics)) as executor:
            futures = {
                executor.submit(self._run_single_topic, t, max_retry, max_revision): t
                for t in topics
            }
            for future in as_completed(futures):
                topic = futures[future]
                result = future.result()
                if result:
                    results_map[topic] = result

        # 원래 토픽 순서 유지
        return "\n\n".join(results_map[t] for t in topics if t in results_map)
