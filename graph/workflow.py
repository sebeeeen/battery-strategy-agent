"""
LangGraph 기반 Competitive Intelligence Multi-Agent 워크플로우
  - supervisor_node: 세 컨텍스트가 모두 비어있을 때 parallel_rag로 직접 라우팅
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

# 독립 클래스 임포트
from agents.lges_rag_agent import LGESRagAgent, LGES_TOPICS
from agents.catl_rag_agent import CATLRagAgent, CATL_TOPICS
from agents.rag_agent import RAGAgent, MARKET_TOPICS
from agents.search_agent import SearchAgent
from agents.critic_agent import CriticAgent
from agents.supervisor import SupervisorAgent
from graph.progress import tracker


# State Definition

class BatteryAnalysisState(TypedDict):
    """LangGraph 공유 상태 — 전체 파이프라인을 통해 누적."""

    # Collected Information
    lges_context: str           # LGESRagAgent 결과
    catl_context: str           # CATLRagAgent 결과
    market_context: str         # Market RAGAgent 결과
    search_results: str         # Web search 결과 (포맷된 텍스트)
    search_sources: List[Dict]  # 검색 소스 메타데이터

    # Competitive Intelligence
    contradictions: List[Dict]  # 탐지된 상충 주장 목록
    coverage_matrix: Dict       # 전략 차원 커버리지 매트릭스

    # Critic Feedback
    critic_feedback: str
    critic_verdict: str         # "APPROVED" | "NEEDS_MORE_SEARCH"
    requery_instructions: List[Dict]

    # Human-in-the-Loop
    human_approved: bool        # 사람의 최종 승인 여부
    human_notes: str            # 사람의 검토 메모

    # Control
    iteration: int
    max_iterations: int
    next_action: str

    # Output
    final_report: str


# Node Functions

def supervisor_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Competitive Intelligence Supervisor Node.

    세 컨텍스트가 모두 비어있을 때 parallel_rag로 즉시 라우팅

    1. 첫 실행: 3개 RAG 에이전트를 병렬로 한번에 디스패치
    2. LGES/CATL 양쪽 데이터 확보 후 상충 주장 탐지
    3. 상충 발견 시 팩트체크 우선 라우팅
    4. Critic 피드백 기반 커버리지 갭 보완
    5. human_review 이후: human_approved=True → generate_report
                         human_approved=True + requery → search (추가 검색 후 재진행)
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    # HiTL 승인 체크
    # human_review를 거쳐 사람이 승인한 경우: Supervisor가 최종 라우팅을 결정
    if state.get("human_approved", False):
        requery = state.get("requery_instructions", [])
        # max_iter 초과 시 requery 무시 → 즉시 generate_report (무한루프 차단)
        if requery and iteration < max_iter:
            tracker.update("supervisor", f"재검색 지시 {len(requery)}건 → search  (반복 {iteration+1}/{max_iter})")
            return {
                **state,
                "next_action": "search",
                "iteration": iteration + 1,
                "human_approved": False,
                "requery_instructions": requery,
            }
        tracker.update("supervisor", "승인 완료 → generate_report")
        return {
            **state,
            "next_action": "generate_report",
            "iteration": iteration + 1,
            "coverage_matrix": state.get("coverage_matrix", {}),
            "requery_instructions": [],   # 잔여 requery 초기화
        }

    agent = SupervisorAgent()

    # 커버리지 매트릭스 업데이트 (데이터가 있을 경우)
    lges = state.get("lges_context", "")
    catl = state.get("catl_context", "")
    market = state.get("market_context", "")
    search = state.get("search_results", "")

    if lges or catl or market or search:
        coverage = agent.assess_coverage(lges, catl, market, search)
    else:
        coverage = state.get("coverage_matrix", {})

    # 세 컨텍스트가 모두 비어있으면 → parallel_rag로 즉시 라우팅
    # parallel_rag → 3개 동시 실행 → supervisor 한 번 복귀
    if not lges and not catl and not market:
        next_action = "parallel_rag"
        requery = None
        tracker.update("supervisor", f"반복 {iteration+1}/{max_iter} → parallel_rag")
    else:
        next_action, requery = agent.decide_next_action(
            lges_context=lges,
            catl_context=catl,
            market_context=market,
            search_results=search,
            critic_verdict=state.get("critic_verdict", ""),
            iteration=iteration,
            max_iterations=max_iter,
        )
        tracker.update("supervisor", f"반복 {iteration+1}/{max_iter} → {next_action}")

    update = {
        **state,
        "next_action": next_action,
        "iteration": iteration + 1,
        "coverage_matrix": coverage,
    }
    if requery:
        update["requery_instructions"] = requery

    return update


# Parallel RAG Node

def parallel_rag_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Parallel RAG Node — asyncio.gather()로 3개 RAG 에이전트 동시 실행

    실행 구조:
      asyncio.gather(lges_task, catl_task, market_task)
        ├── LGESRagAgent().run()   → ThreadPoolExecutor worker-1
        ├── CATLRagAgent().run()   → ThreadPoolExecutor worker-2
        └── RAGAgent(market).run() → ThreadPoolExecutor worker-3

    의존성:
      - 세 에이전트는 서로 독립적 (데이터 격리: faiss_lges / faiss_catl / faiss_market)
      - 병렬 실행으로 처리 시간 약 50~60% 단축
      - 각 에이전트 완료 후 결과를 State에 동시 업데이트

    이벤트 루프 안전 처리:
      - 일반 실행(asyncio.run): 새 이벤트 루프 생성
      - Jupyter / LangGraph 내부 루프 감지 시: 별도 스레드에서 새 루프 실행
    """
    tracker.update("parallel_rag")

    # 동기 함수 정의 (ThreadPoolExecutor에서 실행)
    def run_lges() -> str:
        return LGESRagAgent().run()

    def run_catl() -> str:
        return CATLRagAgent().run()

    def run_market() -> str:
        return RAGAgent(doc_type="market").run(topics=MARKET_TOPICS)

    # asyncio.gather() 코루틴 정의
    async def gather_rag() -> tuple:
        """
        3개 RAG 태스크를 asyncio.gather()로 병렬 실행.

        동기 LLM 호출을 run_in_executor로 비동기 태스크로 래핑.
        """
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=3)
        try:
            lges_task   = loop.run_in_executor(executor, run_lges)
            catl_task   = loop.run_in_executor(executor, run_catl)
            market_task = loop.run_in_executor(executor, run_market)
            # 3개 태스크를 동시 실행 후 모두 완료될 때까지 대기
            return await asyncio.gather(lges_task, catl_task, market_task)
        finally:
            executor.shutdown(wait=False)

    # 이벤트 루프 안전 실행
    try:
        # 이미 실행 중인 이벤트 루프가 있는지 확인
        asyncio.get_running_loop()
        # 실행 중인 루프 감지 → 별도 스레드에서 새 이벤트 루프 생성하여 실행
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as outer_executor:
            results = outer_executor.submit(
                lambda: asyncio.run(gather_rag())
            ).result()
    except RuntimeError:
        # 실행 중인 루프 없음 → 직접 asyncio.run()
        results = asyncio.run(gather_rag())

    lges_result, catl_result, market_result = results

    return {
        **state,
        "lges_context":   lges_result,
        "catl_context":   catl_result,
        "market_context": market_result,
    }


# 개별 RAG 노드 (fallback / 개별 재실행용)

def rag_lges_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """LGESRagAgent — fallback 개별 재실행."""
    return {**state, "lges_context": LGESRagAgent().run()}


def rag_catl_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """CATLRagAgent — fallback 개별 재실행."""
    return {**state, "catl_context": CATLRagAgent().run()}


def rag_market_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """Market RAGAgent — fallback 개별 재실행."""
    return {**state, "market_context": RAGAgent(doc_type="market").run(topics=MARKET_TOPICS)}


def search_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Search Agent — 긍정/부정 쌍 쿼리 + 상충 주장 팩트체크.
    Supervisor가 생성한 팩트체크 쿼리도 함께 처리.
    """
    tracker.update("search")
    agent = SearchAgent()
    requery = state.get("requery_instructions", [])
    result = agent.run_balanced_search(
        additional_queries=requery if requery else None
    )
    search_text = agent.format_for_report(result)
    sources = result.get("sources", [])
    existing_sources = state.get("search_sources", [])

    return {
        **state,
        "search_results": search_text,
        "search_sources": existing_sources + sources,
        "critic_verdict": "",
    }


def critic_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """Critic Agent — 수집 정보 균형성 + 상충 주장 해소 여부 검토."""
    tracker.update("critic")
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


def human_review_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """
    Human-in-the-Loop Review Node.

    HiTL 활성화 시 이 노드에서 실행이 일시정지되고
    사람이 수집된 정보를 검토한 후 승인/수정 지시를 내릴 수 있음.

    LangGraph interrupt 메커니즘:
      compile(interrupt_before=["generate_report"])와 함께 사용 시
      supervisor가 generate_report로 라우팅하기 직전 중단됨.
      app.update_state(config, {...})로 상태 수정 후
      app.stream(None, config)로 재개.

    HiTL 실행 흐름 (v2 — supervisor 경유):
      critic → human_review → supervisor → [interrupt] → generate_report

    사용 예시 (app.py):
        config = {"configurable": {"thread_id": "run-001"}}
        # 스트리밍 실행 (generate_report 직전에 중단)
        for event in app.stream(initial_state, config):
            if "__interrupt__" in str(event):
                print("[HUMAN REVIEW REQUIRED]")
                print(f"Critic: {app.get_state(config).values['critic_feedback']}")
                break
        # 승인 후 재개 (requery_instructions 전달 시 supervisor가 search로 재라우팅)
        app.update_state(config, {
            "human_approved": True,
            "human_notes": "ESS 데이터 보강",
            "requery_instructions": [{"topic": "ESS market share 2025"}],  # 선택적
        })
        for event in app.stream(None, config):
            ...
    """
    tracker.update("human_review")

    critic_feedback = state.get("critic_feedback", "평가 없음")

    # 사람에게 보여줄 핵심 정보만 출력
    verdict_line = critic_feedback.splitlines()[0] if critic_feedback else "N/A"
    print(f"  Critic: {verdict_line}")

    # HiTL 비활성화 모드: 자동 승인
    # max_iterations 도달 시 critic의 requery_instructions를 제거하여
    # supervisor가 search로 재라우팅하는 무한루프 차단
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    human_approved = state.get("human_approved", False)
    if not human_approved:
        human_approved = True

    update = {**state, "human_approved": human_approved}
    if iteration >= max_iter:
        update["requery_instructions"] = []   # 강제 초기화 → supervisor가 generate_report로 라우팅
    return update


def generate_report_node(state: BatteryAnalysisState) -> BatteryAnalysisState:
    """Report Generation Node — Competitive Intelligence 보고서 작성."""
    tracker.update("generate_report")

    if not state.get("human_approved", True):
        return state

    agent = SupervisorAgent()

    # 커버리지 어세스먼트 복원 (Intelligence Gap Report용)
    lges = state.get("lges_context", "")
    catl = state.get("catl_context", "")
    market = state.get("market_context", "")
    search = state.get("search_results", "")
    agent.assess_coverage(lges, catl, market, search)

    report = agent.generate_report(
        lges_context=lges,
        catl_context=catl,
        market_context=market,
        search_results=search,
        sources=state.get("search_sources", []),
        contradictions=state.get("contradictions", []),
    )
    tracker.done()
    return {**state, "final_report": report}


# Routing Functions

def route_from_supervisor(state: BatteryAnalysisState) -> str:
    return state.get("next_action", "generate_report")


def route_after_search(state: BatteryAnalysisState) -> str:
    return "critic"


def route_from_critic(state: BatteryAnalysisState) -> str:
    verdict = state.get("critic_verdict", "APPROVED")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    if verdict == "NEEDS_MORE_SEARCH" and iteration < max_iter:
        return "supervisor"
    return "human_review"   # Critic 통과 후 HiTL 체크포인트로


# Graph Builder

def build_workflow() -> StateGraph:
    """
    LangGraph StateGraph 구성

      - parallel_rag 노드 추가 (asyncio.gather 병렬 실행)
      - supervisor → parallel_rag 조건부 라우팅 추가
      - 개별 rag_lges / rag_catl / rag_market 노드 유지 (개별 재실행 fallback)
    """
    workflow = StateGraph(BatteryAnalysisState)

    # 노드 등록
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("parallel_rag", parallel_rag_node)   # 병렬 RAG
    workflow.add_node("rag_lges", rag_lges_node)           # [fallback] 개별 재실행
    workflow.add_node("rag_catl", rag_catl_node)           # [fallback]
    workflow.add_node("rag_market", rag_market_node)       # [fallback]
    workflow.add_node("search", search_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("generate_report", generate_report_node)

    # 시작점
    workflow.set_entry_point("supervisor")

    # supervisor → 조건부 라우팅
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "parallel_rag": "parallel_rag",  # 초기 병렬 실행
            "rag_lges":    "rag_lges",        # [fallback] 개별 재실행
            "rag_catl":    "rag_catl",
            "rag_market":  "rag_market",
            "search":      "search",
            "generate_report": "generate_report",
        },
    )

    # parallel_rag 완료 → supervisor (3개 결과 한번에 반환)
    workflow.add_edge("parallel_rag", "supervisor")

    # 개별 RAG 노드 → supervisor (fallback 경로)
    workflow.add_edge("rag_lges",   "supervisor")
    workflow.add_edge("rag_catl",   "supervisor")
    workflow.add_edge("rag_market", "supervisor")

    # search → critic
    workflow.add_conditional_edges("search", route_after_search, {"critic": "critic"})

    # critic → (supervisor | human_review)
    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {"supervisor": "supervisor", "human_review": "human_review"},
    )

    # human_review → supervisor (사람의 승인/추가지시를 Supervisor가 최종 라우팅 결정)
    workflow.add_edge("human_review", "supervisor")

    # generate_report → END
    workflow.add_edge("generate_report", END)

    return workflow


def create_app(enable_hitl: bool = False):
    """
    컴파일된 LangGraph 앱 반환.

    Args:
        enable_hitl: Human-in-the-Loop 활성화 여부
            True → interrupt_before=["generate_report"] + MemorySaver 체크포인터
            False → 일반 자동 실행 모드 (기본값)

    HiTL 사용 방법:
        app = create_app(enable_hitl=True)
        config = {"configurable": {"thread_id": "analysis-001"}}

        # 1단계: 실행 (human_review 직전에 중단됨)
        for event in app.stream(initial_state, config):
            pass

        # 2단계: 상태 확인
        state = app.get_state(config).values
        print(state["critic_feedback"])

        # 3단계: 승인 후 재개
        app.update_state(config, {"human_approved": True})
        for event in app.stream(None, config):
            pass
    """
    workflow = build_workflow()

    if enable_hitl:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            print("[Workflow] Human-in-the-Loop mode enabled.")
            print("  → interrupt_before=['generate_report']")
            print("  → 실행 흐름: critic → human_review → supervisor → [중단] → generate_report")
            print("  → MemorySaver checkpointer active")
            return workflow.compile(
                checkpointer=checkpointer,
                interrupt_before=["generate_report"],
            )
        except ImportError:
            print("[Workflow] MemorySaver not available. Falling back to standard mode.")
            return workflow.compile()

    return workflow.compile()


# ─── Initial State Factory ────────────────────────────────────────────────────

def create_initial_state(max_iterations: int = 8) -> BatteryAnalysisState:
    """초기 상태 생성 + ProgressTracker 타이머 시작."""
    tracker.start()
    return BatteryAnalysisState(
        lges_context="",
        catl_context="",
        market_context="",
        search_results="",
        search_sources=[],
        contradictions=[],
        coverage_matrix={},
        critic_feedback="",
        critic_verdict="",
        requery_instructions=[],
        human_approved=False,
        human_notes="",
        iteration=0,
        max_iterations=max_iterations,
        next_action="",
        final_report="",
    )
