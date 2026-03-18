"""
agents/rag_agent.py
RAG Agent A (LGES) / RAG Agent B (CATL) 공통 모듈.

교재 Agentic RAG 워크플로우 구현 :
  Memory Load → Query Transformation → Retrieve
  → Search Evaluation (Grade Documents)
    → NO: Query Rewrite → Retrieve (max_retry: 5)
    → YES: Draft Generation
      → Self-Reflection (APPROVED / REVISE / RETRIEVE)
        → APPROVED: Memory Update → return result
        → REVISE: Draft Revision (max_revision: 3)
        → RETRIEVE: back to Retrieve
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

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


class RAGAgent:
    """
    Agentic RAG Agent — LGES 또는 CATL 문서 기반 정보 추출.

    교재 Self-Reflection 패턴:
    - Draft 생성 후 LLM이 자체 평가 (APPROVED/REVISE/RETRIEVE)
    - max_revision: 3 회 수정 후에도 미흡하면 최선의 결과 반환
    - max_retry: 5 회 재검색 (retriever.py 에서 처리)
    """

    def __init__(self, doc_type: str):
        """
        Args:
            doc_type: "lges" | "catl" | "market"
        """
        self.doc_type = doc_type
        self.company = {
            "lges": "LG에너지솔루션 (LG Energy Solution)",
            "catl": "CATL (Contemporary Amperex Technology Co., Limited)",
            "market": "글로벌 배터리/EV 시장",
        }[doc_type]

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

    def _extract_draft(self, topic: str, context: str) -> str:
        """Draft Generation — 컨텍스트에서 전략 정보 추출."""
        chain = RAG_EXTRACTION_PROMPT | self.llm
        result = chain.invoke({
            "company": self.company,
            "topic": topic,
            "context": context,
        })
        return result.content

    def _self_reflect(self, topic: str, draft: str, context: str) -> dict:
        """
        Self-Reflection — 드래프트 품질 평가.
        Returns: {"verdict": "APPROVED"|"REVISE"|"RETRIEVE", "reason": ..., ...}
        """
        chain = RAG_SELF_REFLECTION_PROMPT | self.llm
        result = chain.invoke({
            "topic": topic,
            "company": self.company,
            "draft": draft,
            "context": context,
        })

        text = result.content.strip()
        # JSON 파싱
        try:
            # 코드 블록 제거
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본 APPROVED 처리
            return {"verdict": "APPROVED", "reason": "JSON parse failed, accepting draft"}

    def _revise_draft(self, topic: str, draft: str, guidance: str, context: str) -> str:
        """Draft Revision — Self-Reflection 피드백 기반 수정."""
        chain = RAG_REVISION_PROMPT | self.llm
        result = chain.invoke({
            "company": self.company,
            "topic": topic,
            "draft": draft,
            "guidance": guidance,
            "context": context,
        })
        return result.content

    def run(
        self,
        topics: List[str],
        max_retry: int = MAX_RETRIEVAL_RETRY,
        max_revision: int = MAX_REVISION,
    ) -> str:
        """
        전체 Agentic RAG 파이프라인 실행.

        Args:
            topics: 분석할 주제 리스트 (예: ["ESS 전략", "포트폴리오 다각화"])
            max_retry: 최대 재검색 횟수 (검색 평가 실패 시)
            max_revision: 최대 드래프트 수정 횟수

        Returns:
            통합된 분석 결과 문자열 (한국어)
        """
        def _run_topic(topic: str) -> str:
            """단일 토픽 RAG 파이프라인 실행 (병렬 처리 단위)."""
            query = f"{self.company} {topic}"
            context = retrieve_context(query, self.doc_type, max_retry=max_retry)
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
                    guidance = reflection.get("revision_guidance", "Improve specificity and data")
                    draft = self._revise_draft(topic, draft, guidance, context)
                    revision_count += 1
                elif verdict == "RETRIEVE":
                    missing = reflection.get("missing_info", "")
                    new_query = f"{self.company} {topic} {missing}"
                    additional_context = retrieve_context(
                        new_query, self.doc_type, max_retry=max_retry
                    )
                    if additional_context:
                        context = context + "\n\n---\n\n" + additional_context
                    draft = self._extract_draft(topic, context)
                    revision_count += 1
                else:
                    break

            return f"## {topic}\n\n{draft}"

        results_map = {}
        with ThreadPoolExecutor(max_workers=len(topics)) as executor:
            futures = {executor.submit(_run_topic, t): t for t in topics}
            for future in as_completed(futures):
                topic = futures[future]
                result = future.result()
                if result:
                    results_map[topic] = result

        # Step 5: Memory Update — 원래 토픽 순서 유지
        return "\n\n".join(results_map[t] for t in topics if t in results_map)


# Predefined Topics per Agent

LGES_TOPICS = [
    "포트폴리오 다각화 전략 및 핵심 사업 방향",
    "ESS(에너지저장장치) 사업 전략",
    "전기차 배터리 시장 대응 전략",
    "주요 경쟁력 및 기술 역량",
    "재무 성과 및 시장 포지셔닝",
]

CATL_TOPICS = [
    "포트폴리오 다각화 전략 및 핵심 사업 방향",
    "ESS(에너지저장장치) 사업 전략",
    "나트륨이온 배터리 및 신기술 전략",
    "글로벌 시장 확장 전략",
    "주요 경쟁력 및 LFP 배터리 강점",
]

MARKET_TOPICS = [
    "글로벌 전기차 및 배터리 시장 현황과 캐즘(Chasm) 현상",
    "ESS 시장 성장 전망",
    "주요 지역별 EV 시장 동향",
]
