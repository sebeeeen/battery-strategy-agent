# Battery Strategy Analysis Multi-Agent System

LG에너지솔루션(LGES)과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 Multi-Agent 보고서 자동 생성 시스템.

## Overview

- **Objective**: 전기차 캐즘 환경 속에서 LGES와 CATL의 포트폴리오 다각화 전략을 객관적 데이터 기반으로 비교 분석
- **Method**: Supervisor 패턴 Multi-Agent + Agentic RAG (CRAG 기반 Self-Reflection)
- **Tools**: LangGraph, LangChain, FAISS, BAAI/bge-m3, Tavily Search

## Features

- PDF 자료 기반 정보 추출 (LGES ESG Report, CATL ESG Report, IEA Global EV Outlook 2024)
- Agentic RAG: Query Transformation → Retrieve → Grade Documents → Query Rewrite (max_retry: 5) → Self-Reflection (max_revision: 3)
- 확증 편향 방지 전략: Search Agent가 각 주제별 긍정 쿼리("LGES ESS 전략 성과")와 부정 쿼리("LGES ESS 전략 한계·문제점")를 쌍으로 실행
- Critic Agent가 수집 결과의 균형성을 독립 검토, 편향 발견 시 Supervisor에게 재검색 지시
- 최종 출력: Markdown + PDF 형식 전략 분석 보고서

## Tech Stack

| Category   | Details                                         |
|------------|-------------------------------------------------|
| Framework  | LangGraph, LangChain, Python 3.11               |
| LLM        | gpt-4.1-mini via OpenAI API                     |
| Retrieval  | FAISS (faiss-cpu)                               |
| Embedding  | BAAI/bge-m3 (오픈소스, 100+ 언어 멀티링컬)       |
| Web Search | Tavily Search API                               |
| PDF Parse  | PyMuPDF (fitz)                                  |
| Output     | Markdown → PDF (weasyprint)                     |

## Agents

- **Supervisor Agent**: 전체 워크플로우 조율, 작업 분배 결정, 최종 보고서 생성
- **RAG Agent A (LGES)**: LGES PDF 문서 벡터 검색 및 전략 컨텍스트 추출 (Agentic RAG)
- **RAG Agent B (CATL)**: CATL PDF 문서 벡터 검색 및 전략 컨텍스트 추출 (Agentic RAG)
- **Search Agent**: 웹 검색 (긍정/부정 쌍 쿼리로 확증 편향 방지)
- **Critic Agent**: 수집 정보 균형성 검토, 편향 발견 시 재검색 지시

## Architecture

### 전체 워크플로우 (v3 — 비동기 병렬 실행)

![전체 워크플로우](./assets/전체%20워크플로우.png)

### 병렬 실행 시퀀스 다이어그램

![병렬 실행 시퀀스](./assets/병렬%20실행%20시퀀스%20다이어그램.png)

### Agentic RAG 내부 플로우

![Agentic RAG 플로우](./assets/Agentic%20RAG%20내부%20플로우.png)

## Directory Structure

```
battery-strategy-agent/
├── .env                        # API 키 (git 제외)
├── config.py                   # 설정값 통합
├── app.py                      # 파이프라인 진입점
│
├── agents/
│   ├── tools.py                # ★ @tool 데코레이터 정의 (5개)
│   ├── lges_rag_agent.py       # ★ LGES 전용 독립 RAG Agent
│   ├── catl_rag_agent.py       # ★ CATL 전용 독립 RAG Agent
│   ├── rag_agent.py            # Market RAG (공통)
│   ├── search_agent.py         # 확증 편향 방지 검색
│   ├── critic_agent.py         # 균형성 독립 평가
│   └── supervisor.py           # ★ Competitive Intelligence Supervisor
│
├── graph/
│   └── workflow.py             # ★ LangGraph StateGraph + HiTL + 비동기 병렬 RAG 
│
├── retrieval/
│   ├── ingest.py               # PDF → FAISS 인덱스 생성
│   └── retriever.py            # CRAG 검색 루프
│
├── prompts/
│   ├── rag.py                  # RAG 프롬프트 3종
│   ├── search.py               # 검색 프롬프트 + 쌍 쿼리
│   ├── critic.py               # Critic 평가 프롬프트
│   └── supervisor.py           # 보고서 생성 프롬프트
│
├── data/pdf/                   # 원본 PDF 3종 (100p)
├── data/faiss_index/           # FAISS 인덱스 (3개)
│   ├── faiss_lges/
│   ├── faiss_catl/
│   └── faiss_market/
│
├── outputs/                    # 생성된 보고서 저장
└── requirements.txt
```

## Key Implementations

### @tool 데코레이터 — OpenAI Function Calling 연동

```python
# agents/tools.py
from langchain_core.tools import tool

@tool
def retrieve_lges_documents(query: str) -> str:
    """LGES 전략 문서에서 관련 정보를 검색합니다."""
    return retrieve_context(query, doc_type="lges")

@tool
def retrieve_catl_documents(query: str) -> str:
    """CATL 전략 문서에서 관련 정보를 검색합니다."""
    return retrieve_context(query, doc_type="catl")
```

```python
# agents/lges_rag_agent.py
self.llm = ChatOpenAI(model="gpt-4.1-mini").bind_tools([retrieve_lges_documents])
```

### 비동기 병렬 RAG 실행 

Parallel Collaboration 기반 — 3개 RAG 에이전트를 `asyncio.gather()`로 동시 실행하여 처리 시간을 약 50~60% 단축한다.

```python
# graph/workflow.py — parallel_rag_node

async def gather_rag() -> tuple:
    """3개 RAG 태스크를 asyncio.gather()로 병렬 실행."""
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        lges_task   = loop.run_in_executor(executor, lambda: LGESRagAgent().run())
        catl_task   = loop.run_in_executor(executor, lambda: CATLRagAgent().run())
        market_task = loop.run_in_executor(executor, lambda: RAGAgent(doc_type="market").run())
        return await asyncio.gather(lges_task, catl_task, market_task)
    finally:
        executor.shutdown(wait=False)

# 이벤트 루프 안전 처리 (Jupyter/LangGraph 환경 호환)
try:
    asyncio.get_running_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as outer_executor:
        results = outer_executor.submit(lambda: asyncio.run(gather_rag())).result()
except RuntimeError:
    results = asyncio.run(gather_rag())
```

### Human-in-the-Loop (HiTL) 실행 흐름

`critic → human_review → supervisor → [interrupt] → generate_report` 흐름으로, Supervisor가 human_review 이후의 최종 라우팅을 결정한다.

```python
# graph/workflow.py
app = create_app(enable_hitl=True)
# → compile(checkpointer=MemorySaver(), interrupt_before=["generate_report"])

config = {"configurable": {"thread_id": "analysis-001"}}

# 1단계: 분석 실행 (supervisor → generate_report 직전 자동 중단)
for event in app.stream(initial_state, config):
    if "__interrupt__" in str(event):
        break

# 2단계: 사람이 Critic 피드백 확인
state = app.get_state(config).values
print(state["critic_feedback"])
print(state["coverage_matrix"])

# 3단계: 승인 또는 추가 검색 지시
# ① 단순 승인
app.update_state(config, {"human_approved": True})

# ② 추가 검색 지시 (requery_instructions 전달 시 Supervisor가 search로 재라우팅)
app.update_state(config, {
    "human_approved": True,
    "human_notes": "ESS 시장 점유율 데이터 보강 필요",
    "requery_instructions": [{"topic": "ESS market share 2025", "focus": "LGES vs CATL"}]
})

# 4단계: 재개 (Supervisor가 requery 여부에 따라 search 또는 generate_report로 라우팅)
for event in app.stream(None, config):
    pass  # 최종 보고서 생성 완료
```

## Setup & Run

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. PDF 변환 시스템 의존성 설치 (weasyprint 권장 — macOS 기준)
brew install cairo pango gdk-pixbuf libffi
pip install weasyprint

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 4. PDF 파일 배치
cp path/to/lges.pdf   data/lges/
cp path/to/catl.pdf   data/catl/
cp path/to/iea.pdf    data/market/

# 5. 문서 인덱싱 (최초 1회)
python retrieval/ingest.py

# 6. 보고서 생성
python app.py
```

## Contributors

- 권세빈: Agentic RAG 설계, RAG Agent 개발, 확증 편향 방지 전략 구현, Critic Agent 개발, LangGraph StateGraph 구성
