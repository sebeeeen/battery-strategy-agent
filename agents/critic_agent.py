"""
agents/critic_agent.py
Critic Agent — 수집 정보의 균형성 독립 검토 및 편향 탐지.

역할:
- RAG 결과 + 웹 검색 결과를 독립적으로 검토
- 긍/부정 균형, LGES/CATL 정보 균형, 커버리지 점검
- 편향 발견 시 재검색 지시 (어떤 주제/방향인지 명시)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from prompts.critic import CRITIC_BALANCE_PROMPT, CRITIC_REQUERY_PROMPT


class CriticAgent:
    """
    독립 검토 Agent — 정보 균형성 및 편향 탐지.

    평가 기준:
    1. 균형성: LGES/CATL 각각 긍정·부정 모두 커버
    2. 대칭성: 두 회사 분석 깊이가 유사
    3. 데이터 품질: 구체적 수치/사실 포함
    4. 커버리지: 시장 배경, 전략, 비교 분석 모두 포함
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.0,  # 판단의 일관성을 위해 temperature=0
            openai_api_key=OPENAI_API_KEY,
        )

    def _truncate(self, text: str, max_chars: int = 1500) -> str:
        """LLM 컨텍스트 절약을 위한 텍스트 자르기."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n...(truncated, total {len(text)} chars)"

    def evaluate(
        self,
        lges_context: str,
        catl_context: str,
        market_context: str,
        search_results: str,
    ) -> Dict[str, Any]:
        """
        수집된 모든 정보의 균형성과 완전성 평가.

        Returns:
            {
                "is_balanced": bool,
                "balance_score": float,
                "issues": List[str],
                "missing_topics": List[str],
                "requery_instructions": List[Dict],
                "verdict": "APPROVED" | "NEEDS_MORE_SEARCH"
            }
        """
        print("\n[Critic Agent] Evaluating information balance...")

        chain = CRITIC_BALANCE_PROMPT | self.llm
        result = chain.invoke({
            "lges_context": self._truncate(lges_context),
            "catl_context": self._truncate(catl_context),
            "market_context": self._truncate(market_context),
            "search_results": self._truncate(search_results),
        })

        text = result.content.strip()

        # JSON 파싱
        try:
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]
            evaluation = json.loads(text)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값: 통과
            evaluation = {
                "is_balanced": True,
                "balance_score": 0.7,
                "issues": ["JSON parse error in critic evaluation"],
                "missing_topics": [],
                "requery_instructions": [],
                "verdict": "APPROVED",
            }

        verdict = evaluation.get("verdict", "APPROVED")
        score = evaluation.get("balance_score", 0.0)
        issues = evaluation.get("issues", [])

        print(f"  Verdict: {verdict} (score: {score:.2f})")
        if issues:
            print(f"  Issues: {'; '.join(issues[:3])}")

        return evaluation

    def get_requery_instructions(
        self,
        issues: List[str],
        missing_topics: List[str],
    ) -> List[Dict[str, str]]:
        """
        발견된 이슈에 대한 재검색 지시 생성.

        Returns:
            [{"topic": "...", "query": "...", "query_type": "positive|negative"}, ...]
        """
        if not issues and not missing_topics:
            return []

        chain = CRITIC_REQUERY_PROMPT | self.llm
        result = chain.invoke({
            "issues": "\n".join(f"- {i}" for i in issues),
            "missing_topics": "\n".join(f"- {t}" for t in missing_topics),
        })

        text = result.content.strip()
        try:
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError:
            return []
