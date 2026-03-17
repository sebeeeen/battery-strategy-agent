"""
agents/supervisor.py
Supervisor Agent — 전체 워크플로우 조율, 작업 분배, 보고서 생성.

Supervisor 패턴:
- 현재 State를 보고 다음 액션을 결정
- 각 Sub-agent(RAG A/B, Search, Critic)를 순서에 맞게 호출
- 모든 정보가 모이면 최종 보고서 생성
"""

from pathlib import Path
from typing import Optional
from datetime import date

from langchain_openai import ChatOpenAI

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE_REPORT
from prompts.supervisor import REPORT_GENERATION_PROMPT


class SupervisorAgent:
    """
    Supervisor Agent — 보고서 최종 생성 담당.

    LangGraph workflow.py 의 StateGraph에서 조율 로직을 담당하며,
    이 클래스는 최종 보고서 생성 역할에 집중.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE_REPORT,
            openai_api_key=OPENAI_API_KEY,
        )

    def generate_report(
        self,
        lges_context: str,
        catl_context: str,
        market_context: str,
        search_results: str,
        sources: Optional[list] = None,
    ) -> str:
        """
        수집된 모든 정보를 통합하여 최종 보고서 생성.

        Args:
            lges_context: RAG Agent A 결과 (LGES 분석)
            catl_context: RAG Agent B 결과 (CATL 분석)
            market_context: 시장 배경 RAG 결과
            search_results: 웹 검색 결과 요약
            sources: 참고 자료 목록

        Returns:
            Markdown 형식의 보고서 문자열
        """
        print("\n[Supervisor] Generating final report...")

        # 소스 목록 포맷
        source_list = self._format_sources(sources or [])

        chain = REPORT_GENERATION_PROMPT | self.llm
        result = chain.invoke({
            "lges_context": lges_context,
            "catl_context": catl_context,
            "market_context": market_context,
            "search_results": search_results,
            "source_list": source_list,
        })

        report = result.content

        # 보고서 헤더 추가
        today = date.today().strftime("%Y년 %m월 %d일")
        header = (
            f"# 배터리 시장 전략 분석 보고서\n"
            f"**LGES vs CATL 포트폴리오 다각화 전략 비교 분석**\n\n"
            f"작성일: {today} | 작성 방식: Multi-Agent (LangGraph)\n\n"
            f"---\n\n"
        )

        return header + report

    def _format_sources(self, sources: list) -> str:
        """소스 목록을 REFERENCE 형식으로 포맷."""
        if not sources:
            return "(소스 목록 없음)"

        lines = []
        # 기관 보고서
        report_sources = [s for s in sources if s.get("type") == "report"]
        if report_sources:
            lines.append("**기관 보고서:**")
            for s in report_sources:
                lines.append(f"- {s.get('org', '')}({s.get('year', '')}). {s.get('title', '')}. {s.get('url', '')}")

        # 웹페이지
        web_sources = [s for s in sources if s.get("type") != "report"]
        if web_sources:
            lines.append("**웹페이지:**")
            for s in web_sources:
                title = s.get("title", s.get("url", ""))
                url = s.get("url", "")
                lines.append(f"- {title}. {url}")

        return "\n".join(lines)
