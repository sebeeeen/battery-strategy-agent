"""
agents/search_agent.py
Web Search Agent — 확증 편향 방지를 위한 긍정/부정 쌍 쿼리 전략.

핵심 설계:
- 각 주제에 대해 긍정 쿼리(성과·기회) + 부정 쿼리(한계·위협)를 쌍으로 실행
- 두 결과를 모두 수집하여 Critic Agent에 균형 잡힌 데이터 제공
- Tavily Search API 사용
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OPENAI_API_KEY, TAVILY_API_KEY,
    LLM_MODEL, LLM_TEMPERATURE,
    SEARCH_MAX_RESULTS,
)
from prompts.search import (
    BALANCED_QUERY_PAIRS,
    SEARCH_QUERY_GEN_PROMPT,
    SEARCH_SUMMARIZE_PROMPT,
)


class SearchAgent:
    """
    Dual-Query Web Search Agent.

    확증 편향 방지 전략:
    - 긍정 쿼리: 성과, 강점, 기회 중심
    - 부정 쿼리: 한계, 위험, 비판 중심
    - 두 쿼리 쌍의 결과를 함께 반환
    """

    def __init__(self):
        import os
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

        self.search_tool = TavilySearchResults(
            max_results=SEARCH_MAX_RESULTS,
            search_depth="advanced",
            include_answer=True,
        )
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

    def _run_single_search(self, query: str) -> List[Dict[str, Any]]:
        """단일 쿼리 검색 실행."""
        try:
            results = self.search_tool.invoke(query)
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"  [Search] Error: {e}")
            return []

    def _summarize_results(
        self,
        topic: str,
        query_type: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """검색 결과를 LLM으로 요약."""
        if not results:
            return f"[{query_type} 검색 결과 없음]"

        results_text = "\n".join([
            f"- [{r.get('url', 'N/A')}] {r.get('content', '')[:400]}"
            for r in results
        ])

        chain = SEARCH_SUMMARIZE_PROMPT | self.llm
        result = chain.invoke({
            "topic": topic,
            "query_type": query_type,
            "results": results_text,
        })
        return result.content

    def run_balanced_search(
        self,
        query_pairs: Optional[List[Dict]] = None,
        additional_queries: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        긍정/부정 쌍 쿼리로 균형 잡힌 검색 실행.

        Args:
            query_pairs: 커스텀 쿼리 쌍 (없으면 BALANCED_QUERY_PAIRS 사용)
            additional_queries: Critic Agent가 요청한 추가 검색

        Returns:
            {
                "summaries": {"topic": {"positive": str, "negative": str}, ...},
                "raw_results": [...],
                "sources": [{"url": ..., "title": ...}, ...]
            }
        """
        pairs = query_pairs or BALANCED_QUERY_PAIRS
        if additional_queries:
            # Critic이 요청한 추가 쿼리를 쌍 형태로 변환
            for aq in additional_queries:
                topic = aq.get("topic", "")
                q = aq.get("query", "")
                qtype = aq.get("query_type", "positive")
                # 기존 토픽이 있으면 해당 쌍에 추가, 없으면 새 쌍 생성
                existing = next((p for p in pairs if p["topic"] == topic), None)
                if existing:
                    existing[qtype] = q
                else:
                    pairs.append({"topic": topic, qtype: q})

        summaries = {}
        raw_results = []
        sources = []

        print(f"\n[Search Agent] Running {len(pairs)} balanced query pairs...")

        for pair in pairs:
            topic = pair.get("topic", "unknown")
            pos_query = pair.get("positive", "")
            neg_query = pair.get("negative", "")

            print(f"  Topic: {topic}")

            pos_results, neg_results = [], []

            # 긍정 쿼리
            if pos_query:
                print(f"    [+] {pos_query[:60]}...")
                pos_results = self._run_single_search(pos_query)
                raw_results.extend(pos_results)
                for r in pos_results:
                    if r.get("url"):
                        sources.append({"url": r["url"], "title": r.get("title", r["url"])})

            # 부정 쿼리
            if neg_query:
                print(f"    [-] {neg_query[:60]}...")
                neg_results = self._run_single_search(neg_query)
                raw_results.extend(neg_results)
                for r in neg_results:
                    if r.get("url"):
                        sources.append({"url": r["url"], "title": r.get("title", r["url"])})

            # 요약
            summaries[topic] = {
                "positive": self._summarize_results(topic, "긍정적 측면", pos_results),
                "negative": self._summarize_results(topic, "부정적/비판적 측면", neg_results),
            }

        # 중복 소스 제거
        seen_urls = set()
        unique_sources = []
        for s in sources:
            if s["url"] not in seen_urls:
                seen_urls.add(s["url"])
                unique_sources.append(s)

        return {
            "summaries": summaries,
            "raw_results": raw_results,
            "sources": unique_sources,
        }

    def format_for_report(self, search_result: Dict[str, Any]) -> str:
        """검색 결과를 보고서용 텍스트로 포맷."""
        lines = []
        for topic, data in search_result.get("summaries", {}).items():
            lines.append(f"### {topic}")
            if data.get("positive"):
                lines.append(f"**긍정적 관점:**\n{data['positive']}")
            if data.get("negative"):
                lines.append(f"**비판적 관점:**\n{data['negative']}")
            lines.append("")
        return "\n".join(lines)
