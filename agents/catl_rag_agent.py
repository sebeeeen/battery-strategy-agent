"""
CATL(Contemporary Amperex Technology Co., Limited) 전용 RAG Agent

분리 근거:
  - 독립된 FAISS 인덱스 참조 (faiss_catl) → LGES 문서와 혼용 불가
  - CATL 전용 시스템 프롬프트 (중국 배터리 기업 특화 분석 관점)
  - @tool 기반 retrieve_catl_documents 전용 도구 바인딩
  - 독립된 LLM 인스턴스 → 상태 격리
  - CATL 특화 토픽 목록 (나트륨이온, LFP, 글로벌 확장 등)

교재 Agentic RAG 워크플로우 (완전 구현):
  Query Transformation → Retrieve (CATL FAISS만)
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
from agents.tools import retrieve_catl_documents


# CATL 전용 토픽
CATL_TOPICS = [
    "포트폴리오 다각화 전략 및 핵심 사업 방향",
    "ESS(에너지저장장치) 사업 전략",
    "주요 경쟁력 및 LFP · 나트륨이온 기술 강점",
]


# CATLRagAgent

class CATLRagAgent:
    """
    CATL 전용 Agentic RAG Agent.

    독립 클래스 설계 원칙:
    - CATL FAISS 인덱스만 접근 (faiss_catl 디렉토리)
    - LGES/Market 인덱스와 물리적으로 분리
    - retrieve_catl_documents @tool 바인딩으로 LLM 자율 검색 지원
    - CATL 기업 컨텍스트 특화 프롬프트
      (중국 제조업 배경, LFP 우위, 나트륨이온 혁신, 해외 확장 전략)

    LGESRagAgent와의 차이점:
    - DOC_TYPE: "catl" (vs "lges")
    - COMPANY_NAME: CATL 공식 명칭
    - 전용 @tool: retrieve_catl_documents
    - CATL_TOPICS: LGES와 다른 전략 차원 포함 (나트륨이온, LFP 강점 등)
    """

    # 클래스 고유 식별자
    DOC_TYPE = "catl"
    COMPANY_NAME = "CATL (Contemporary Amperex Technology Co., Limited)"
    INDEX_NAME = "faiss_catl"

    def __init__(self):
        # LLM 인스턴스 — LGESRagAgent와 완전히 분리된 독립 인스턴스
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        # @tool 바인딩 — CATL 전용 검색 도구만 바인딩
        self.llm_with_tools = self.llm.bind_tools([retrieve_catl_documents])

        self._log_prefix = "[CATLRagAgent]"

    def _log(self, msg: str):
        print(f"{self._log_prefix} {msg}")

    # 핵심 메서드

    def _transform_query(self, topic: str) -> str:
        """Query Transformation — CATL 특화 쿼리로 변환."""
        return f"{self.COMPANY_NAME} {topic}"

    def _retrieve(self, query: str, max_retry: int) -> str:
        """
        Retrieve (CATL 인덱스 전용) + Grade Documents.
        doc_type="catl"로 고정 — LGES 인덱스 접근 물리적 불가.
        """
        return retrieve_context(
            query,
            doc_type=self.DOC_TYPE,   # "catl" — LGES 인덱스 접근 불가
            max_retry=max_retry,
        )

    def _extract_draft(self, topic: str, context: str) -> str:
        """Draft Generation — CATL 컨텍스트에서 전략 정보 추출."""
        chain = RAG_EXTRACTION_PROMPT | self.llm
        result = chain.invoke({
            "company": self.COMPANY_NAME,
            "topic": topic,
            "context": context,
        })
        return result.content

    def _self_reflect(self, topic: str, draft: str, context: str) -> dict:
        """
        Self-Reflection — CATL 드래프트 품질 자체 평가.
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

    # 공개 메서드

    def _run_single_topic(self, topic: str, max_retry: int, max_revision: int) -> str:
        """단일 토픽 RAG 파이프라인 실행 (병렬 처리 단위)."""
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
        CATL 전용 Agentic RAG 파이프라인 실행.

        토픽별 병렬 처리 (ThreadPoolExecutor):
          topic-1 ┐
          topic-2 ├── 동시 실행 → 합산
          topic-3 ┘
        """
        topics = topics or CATL_TOPICS
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

        return "\n\n".join(results_map[t] for t in topics if t in results_map)
