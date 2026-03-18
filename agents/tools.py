"""
LangChain @tool 데코레이터 기반 도구 정의.

@tool 데코레이터 성능 이점:
  1. JSON Schema 자동 생성 → llm.bind_tools()로 OpenAI Function Calling 활성화
  2. LLM이 "언제 검색할지" 자율 결정 (Agentic 행동 강화)
  3. 입력 유효성 검사 및 오류 처리 내장
  4. 비동기(arun) 지원으로 병렬 검색 가능
  5. LangSmith 추적 시 도구 호출 단계 가시화
"""

import os
import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import TAVILY_API_KEY, SEARCH_MAX_RESULTS


# RAG Retrieval Tools

@tool
def retrieve_lges_documents(query: str) -> str:
    """
    Args:
        query: 검색할 질문 또는 주제 (예: "LGES ESS 사업 전략", "포트폴리오 다각화")

    Returns:
        관련 문서 청크들을 하나의 문자열로 합친 컨텍스트
    """
    from retrieval.retriever import retrieve_context
    return retrieve_context(query, doc_type="lges")


@tool
def retrieve_catl_documents(query: str) -> str:
    """
    Args:
        query: 검색할 질문 또는 주제 (예: "CATL LFP 배터리 전략", "나트륨이온 배터리 개발")

    Returns:
        관련 문서 청크들을 하나의 문자열로 합친 컨텍스트
    """
    from retrieval.retriever import retrieve_context
    return retrieve_context(query, doc_type="catl")


@tool
def retrieve_market_documents(query: str) -> str:
    """
    Args:
        query: 검색할 시장 관련 질문 (예: "글로벌 EV 시장 성장률", "ESS 시장 전망 2024")

    Returns:
        관련 시장 데이터 청크들을 하나의 문자열로 합친 컨텍스트
    """
    from retrieval.retriever import retrieve_context
    return retrieve_context(query, doc_type="market")


# Web Search Tools

@tool
def search_positive_perspective(topic: str, query: str) -> str:
    """
    Args:
        topic: 분석 주제 (예: "LGES ESS 사업", "CATL 글로벌 확장")
        query: 구체적인 긍정 관점 검색어

    Returns:
        검색 결과 요약 문자열
    """
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults(
        max_results=SEARCH_MAX_RESULTS,
        search_depth="advanced",
        include_answer=True,
    )
    try:
        results = search.invoke(query)
        if not results:
            return f"[{topic}] 긍정 관점 검색 결과 없음"
        formatted = "\n".join([
            f"- [{r.get('url', '')}] {r.get('content', '')[:300]}"
            for r in results
        ])
        return f"[{topic} — 긍정적 관점]\n{formatted}"
    except Exception as e:
        return f"[{topic}] 검색 오류: {e}"


@tool
def search_critical_perspective(topic: str, query: str) -> str:
    """
    특정 주제에 대한 비판적 관점(한계·위험·약점)을 웹에서 검색합니다.

    확증 편향 방지 전략의 부정 쿼리 역할:
    최신 분석 보고서, 비판적 기사에서 해당 주제의
    한계, 위험 요소, 경쟁 압력에 관한 정보를 수집합니다.

    Args:
        topic: 분석 주제 (예: "LGES ESS 사업", "CATL 기술 리스크")
        query: 구체적인 비판/부정 관점 검색어

    Returns:
        검색 결과 요약 문자열
    """
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults(
        max_results=SEARCH_MAX_RESULTS,
        search_depth="advanced",
        include_answer=True,
    )
    try:
        results = search.invoke(query)
        if not results:
            return f"[{topic}] 비판적 관점 검색 결과 없음"
        formatted = "\n".join([
            f"- [{r.get('url', '')}] {r.get('content', '')[:300]}"
            for r in results
        ])
        return f"[{topic} — 비판적 관점]\n{formatted}"
    except Exception as e:
        return f"[{topic}] 검색 오류: {e}"


@tool
def resolve_strategic_contradiction(
    topic: str,
    lges_claim: str,
    catl_claim: str,
) -> str:
    """
    LGES와 CATL의 상충하는 전략적 주장을 최신 시장 데이터로 검증합니다.

    두 기업이 같은 영역에서 상반된 주장(예: 양사 모두 ESS 시장 1위 주장)을
    할 경우, 독립적인 제3자 시장 데이터를 검색하여 사실 관계를 확인합니다.

    Args:
        topic: 상충하는 전략 영역 (예: "ESS 시장 점유율", "LFP vs NMC 우위")
        lges_claim: LGES의 주장 요약
        catl_claim: CATL의 주장 요약

    Returns:
        제3자 시장 데이터 및 팩트체크 결과
    """
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults(
        max_results=SEARCH_MAX_RESULTS,
        search_depth="advanced",
        include_answer=True,
    )
    fact_query = (
        f"{topic} market data independent analysis 2024 2025 "
        f"LGES CATL comparison third party report"
    )
    try:
        results = search.invoke(fact_query)
        if not results:
            return f"[{topic}] 검증 데이터 없음"
        formatted = "\n".join([
            f"- [{r.get('url', '')}] {r.get('content', '')[:350]}"
            for r in results
        ])
        return (
            f"[전략 상충 검증: {topic}]\n"
            f"LGES 주장: {lges_claim}\n"
            f"CATL 주장: {catl_claim}\n"
            f"독립 검증 데이터:\n{formatted}"
        )
    except Exception as e:
        return f"[{topic}] 검증 검색 오류: {e}"


# Tool Collections

# RAG 도구 — 각 에이전트가 bind_tools()에 사용
LGES_TOOLS = [retrieve_lges_documents]
CATL_TOOLS = [retrieve_catl_documents]
MARKET_TOOLS = [retrieve_market_documents]

# 검색 도구 — Search Agent가 bind_tools()에 사용
SEARCH_TOOLS = [
    search_positive_perspective,
    search_critical_perspective,
    resolve_strategic_contradiction,
]

# 전체 도구 목록
ALL_TOOLS = LGES_TOOLS + CATL_TOOLS + MARKET_TOOLS + SEARCH_TOOLS
