"""
graph/workflow.py
LangGraph 기반 Supervisor 패턴 Multi-Agent 워크플로우.

아키텍처:
  START
    │
    ▼
  supervisor_node  ──────────────────────────────────────────┐
    │ (라우팅 결정)                                           │
    ├─── "rag_lges"   ──► rag_lges_node   ──► supervisor_node │
    ├─── "rag_catl"   ──► rag_catl_node   ──► supervisor_node │
    ├─── "rag_market" ──► rag_market_node ──► supervisor_node │
    ├─── "search"     ──► search_node     ──► critic_node     │
    │                                           │             │
    │                                           ▼             │
    │                                     critic_node         │
    │                                           │             │
    │                        ┌─── APPROVED ────►│             │
    │                        │    (or max iter) │             │
    │                        └─── NEEDS_MORE ──►supervisor────┘
    │
    └─── "generate_report" ──► generate_report_node ──► END
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from agents.rag_agent import (
    RAGAgent,
    LGES_TOPICS, CATL_TOPICS, MARKET_TOPICS,
)
from agents.search_agent import SearchAgent
from agents.critic_agent import CriticAgent
from agents.supervisor import SupervisorAgent


# ─── State Definition ─────────────────────────────────────────────────────────

class BatteryAnalysisState(TypedDict):
    """LangGraph 공유 상태 — 전체 파이프라인을 통해 누적."""

    # Collected Information
    lges_context: str           # RAG Agent A 결과
    catl_context: str           # RAG Agent B 결과
    market_context: str         # Market RAG 결과
    search_results: str         # Web search 결과 (포맷된 텍스트)
    search_sources: List[Dict]  # 검색 소스 메타데이터

    # Critic Feedback
    critic_feedback: str        # Critic 평가 요약
    critic_verdict: str         # "APPROVED" | "NEEDS_MORE_SEARCH"
    requery_instructions: List[Dict]  # 재검색 지시

    # Control
    iteration: int              # 현재 반복 횟수
    max_iterations: int         # 최대 반복 횟수
    next_action: str            # 다음 액션

    # Output
    final_report: str           # 최종 보고서 (Markdown)


# ─── Node Functions ───────────────────────────────────────────────────────────

def supervisor_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Supervisor Node — 현재 상태를 보고 다음 액션 결정.
    모든 데이터 수집 완료 여부를 체크하고 적절한 다음 단계 지시.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    # 최대 반복 초과 시 강제 보고서 생성
    if iteration >= max_iter:
        print(f"[Supervisor] Max iterations ({max_iter}) reached. Generating report.")
        return {**state, "next_action": "generate_report"}

    lges = state.get("lges_context", "")
    catl = state.get("catl_context", "")
    market = state.get("market_context", "")
    search = state.get("search_results", "")
    critic_verdict = state.get("critic_verdict", "")

    # 순서대로 필요한 것 처리
    if not lges:
        next_action = "rag_lges"
    elif not catl:
        next_action = "rag_catl"
    elif not market:
        next_action = "rag_market"
    elif not search:
        next_action = "search"
    elif not critic_verdict:
        # critic은 search 노드 후 자동 호출됨 (routing에서 처리)
        next_action = "generate_report"
    elif critic_verdict == "NEEDS_MORE_SEARCH":
        next_action = "search"
    else:
        next_action = "generate_report"

    print(f"[Supervisor] Iteration {iteration+1}/{max_iter}: next_action = {next_action}")
    return {**state, "next_action": next_action, "iteration": iteration + 1}


def rag_lges_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """RAG Agent A — LGES PDF 문서에서 전략 정보 추출."""
    print("\n[RAG-LGES Node] Starting LGES analysis...")
    agent = RAGAgent(doc_type="lges")
    result = agent.run(topics=LGES_TOPICS)
    return {**state, "lges_context": result}


def rag_catl_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """RAG Agent B — CATL PDF 문서에서 전략 정보 추출."""
    print("\n[RAG-CATL Node] Starting CATL analysis...")
    agent = RAGAgent(doc_type="catl")
    result = agent.run(topics=CATL_TOPICS)
    return {**state, "catl_context": result}


def rag_market_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """RAG Market — IEA 보고서에서 시장 배경 정보 추출."""
    print("\n[RAG-Market Node] Starting market analysis...")
    agent = RAGAgent(doc_type="market")
    result = agent.run(topics=MARKET_TOPICS)
    return {**state, "market_context": result}


def search_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Search Agent — 긍정/부정 쌍 쿼리로 균형 잡힌 웹 검색.
    Critic이 지시한 추가 검색 쿼리도 처리.
    """
    print("\n[Search Node] Running balanced web search...")
    agent = SearchAgent()

    requery = state.get("requery_instructions", [])
    result = agent.run_balanced_search(additional_queries=requery if requery else None)

    search_text = agent.format_for_report(result)
    sources = result.get("sources", [])

    # 기존 소스에 추가
    existing_sources = state.get("search_sources", [])
    all_sources = existing_sources + sources

    return {
        **state,
        "search_results": search_text,
        "search_sources": all_sources,
        "critic_verdict": "",   # 재검색 시 critic 초기화
    }


def critic_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Critic Agent — 수집 정보 균형성 검토.
    편향 발견 시 재검색 지시 생성.
    """
    print("\n[Critic Node] Evaluating information balance...")
    agent = CriticAgent()

    evaluation = agent.evaluate(
        lges_context=state.get("lges_context", ""),
        catl_context=state.get("catl_context", ""),
        market_context=state.get("market_context", ""),
        search_results=state.get("search_results", ""),
    )

    verdict = evaluation.get("verdict", "APPROVED")
    issues = evaluation.get("issues", [])
    missing = evaluation.get("missing_topics", [])
    requery = evaluation.get("requery_instructions", [])

    critic_feedback = (
        f"Verdict: {verdict}\n"
        f"Balance Score: {evaluation.get('balance_score', 0):.2f}\n"
        f"Issues: {'; '.join(issues) if issues else 'None'}\n"
        f"Missing: {'; '.join(missing) if missing else 'None'}"
    )

    return {
        **state,
        "critic_verdict": verdict,
        "critic_feedback": critic_feedback,
        "requery_instructions": requery,
    }


def generate_report_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Report Generation Node — Supervisor가 최종 보고서 작성.
    """
    print("\n[Generate Report Node] Creating final report...")
    agent = SupervisorAgent()

    report = agent.generate_report(
        lges_context=state.get("lges_context", ""),
        catl_context=state.get("catl_context", ""),
        market_context=state.get("market_context", ""),
        search_results=state.get("search_results", ""),
        sources=state.get("search_sources", []),
    )

    return {**state, "final_report": report}


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_from_supervisor(state: BatteryAnalysisState) -> str:
    """Supervisor의 next_action을 기반으로 다음 노드 결정."""
    action = state.get("next_action", "generate_report")
    return action


def route_after_search(state: BatteryAnalysisState) -> str:
    """Search 완료 후 Critic으로 이동."""
    return "critic"


def route_from_critic(state: BatteryAnalysisState) -> str:
    """Critic 결과에 따라 재검색 또는 보고서 생성."""
    verdict = state.get("critic_verdict", "APPROVED")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    if verdict == "NEEDS_MORE_SEARCH" and iteration < max_iter:
        return "supervisor"  # Supervisor가 다시 판단
    return "generate_report"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    LangGraph StateGraph 구성 및 반환.

    그래프 구조:
      supervisor → (rag_lges | rag_catl | rag_market | search | generate_report)
      rag_* → supervisor
      search → critic
      critic → (supervisor | generate_report)
      generate_report → END
    """
    workflow = StateGraph(BatteryAnalysisState)

    # 노드 등록
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag_lges", rag_lges_node)
    workflow.add_node("rag_catl", rag_catl_node)
    workflow.add_node("rag_market", rag_market_node)
    workflow.add_node("search", search_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("generate_report", generate_report_node)

    # 시작점
    workflow.set_entry_point("supervisor")

    # 엣지: supervisor에서 조건부 라우팅
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag_lges": "rag_lges",
            "rag_catl": "rag_catl",
            "rag_market": "rag_market",
            "search": "search",
            "generate_report": "generate_report",
        },
    )

    # RAG 노드 → supervisor로 복귀
    workflow.add_edge("rag_lges", "supervisor")
    workflow.add_edge("rag_catl", "supervisor")
    workflow.add_edge("rag_market", "supervisor")

    # search → critic (항상)
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {"critic": "critic"},
    )

    # critic → (supervisor | generate_report)
    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "supervisor": "supervisor",
            "generate_report": "generate_report",
        },
    )

    # generate_report → END
    workflow.add_edge("generate_report", END)

    return workflow


def create_app():
    """컴파일된 LangGraph 앱 반환."""
    workflow = build_workflow()
    return workflow.compile()


# ─── Initial State Factory ────────────────────────────────────────────────────

def create_initial_state(max_iterations: int = 8) -> BatteryAnalysisState:
    """초기 상태 생성."""
    return BatteryAnalysisState(
        lges_context="",
        catl_context="",
        market_context="",
        search_results="",
        search_sources=[],
        critic_feedback="",
        critic_verdict="",
        requery_instructions=[],
        iteration=0,
        max_iterations=max_iterations,
        next_action="",
        final_report="",
    )
