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

## 설계 대비 변경사항

설계 문서(`DESIGN_REPORT.md`) 기준으로 실제 구현 과정에서 변경된 내용을 아래에 정리합니다.

### 1. 파라미터 조정 (성능 최적화)

| 파라미터 | 설계값 | 구현값 | 변경 이유 |
|---------|--------|--------|----------|
| `RETRIEVAL_K` | 6 | 4 | LLM 호출 비용·시간 절감 |
| `MAX_RETRIEVAL_RETRY` | 5 | 1 | 재검색 루프로 인한 지연 방지 |
| `MAX_REVISION` | 3 | 1 | Self-Reflection 반복 시간 단축 |
| Search query pairs | 5쌍 | 3쌍 | 중복 주제 통합 (ESS·기술 차별화 등) |

파라미터 축소는 기능 제거가 아니라 **실행 가능한 시간 범위 내 동작** 보장을 위한 조정입니다. 설계에 명시된 CRAG 루프, Self-Reflection, Grade Documents 등 핵심 워크플로우는 모두 유지됩니다.

### 2. 배치 Grading 도입 (`retrieval/retriever.py`)

설계에서는 문서 k개를 개별 LLM 호출로 각각 평가(`grade_document()`)했으나, 실제 구현에서는 k개 문서를 **단일 LLM 호출로 일괄 평가**하는 `batch_grade_documents()`로 교체했습니다.

- 변경 전: k회 LLM 호출 (k=4일 때 4회)
- 변경 후: 1회 LLM 호출로 전체 문서 평가 → **약 75% 호출 감소**

### 3. 에이전트 내부 토픽 병렬 처리 추가

설계 문서에는 에이전트 간 병렬 처리(`asyncio.gather()`)만 명시되어 있었으나, 실제 구현에서는 각 RAG 에이전트 내부에서도 **토픽별 `ThreadPoolExecutor` 병렬 처리**를 추가했습니다.

- `LGESRagAgent.run()`, `CATLRagAgent.run()`, `RAGAgent.run()` 모두 적용
- 토픽 수가 3개일 때 기존 순차 처리 대비 약 3배 속도 향상

### 4. ProgressTracker 신규 도입 (`graph/progress.py`)

설계 문서에 없던 컴포넌트입니다. 전체 6단계 워크플로우의 진행 상황을 아래 형식으로 출력합니다.

```
[Step N/6] 단계명  |  세부 내용  |  진행률 Z%  (경과시간s 경과)
```

실행 중 사용자가 현재 어느 단계인지 파악할 수 있도록 가시성을 높이기 위해 추가했습니다.

### 5. 무한루프 차단 로직 추가 (`graph/workflow.py`)

Critic Agent가 `NEEDS_MORE_SEARCH` 판정을 반복 반환할 경우, `supervisor_node → search → critic → human_review → supervisor_node` 루프가 무한 반복되는 문제가 실행 중 발견되었습니다.

설계에는 `max_iterations` 초과 시 동작이 명확히 정의되지 않았으나, 다음 두 가지 안전장치를 추가했습니다.

1. **`supervisor_node`**: `human_approved=True + requery` 분기에 `iteration < max_iter` 조건 추가 → 초과 시 `generate_report`로 강제 라우팅
2. **`human_review_node`**: `iteration >= max_iter` 시 `requery_instructions = []` 강제 초기화 (이중 방어)

### 6. RAG Agent 클래스 분리 (교수님 피드백 반영)

초기에는 `RAGAgent(doc_type="lges")`와 `RAGAgent(doc_type="catl")`처럼 파라미터만 다른 동일 클래스를 사용했으나, 교수님 피드백("데이터 혼합 위험")에 따라 `LGESRagAgent`와 `CATLRagAgent`를 **완전히 독립된 클래스**로 분리했습니다.

- 각 클래스가 고유의 `DOC_TYPE`, `COMPANY_NAME`, `INDEX_NAME`을 클래스 변수로 보유
- 전용 `@tool` 바인딩(`retrieve_lges_documents` / `retrieve_catl_documents`)으로 FAISS 인덱스 접근을 물리적으로 분리

---

## Contributors

- 권세빈: Agentic RAG 설계, RAG Agent 개발, 확증 편향 방지 전략 구현, Critic Agent 개발, LangGraph StateGraph 구성
