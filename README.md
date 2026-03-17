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

```
START
  │
  ▼
Supervisor ──────────────────────────────────────────┐
  │ (라우팅 결정)                                     │
  ├── rag_lges   ──► RAG Agent A (LGES) ──► Supervisor│
  ├── rag_catl   ──► RAG Agent B (CATL) ──► Supervisor│
  ├── rag_market ──► RAG Agent M (IEA)  ──► Supervisor│
  └── search ──► Search Agent ──► Critic Agent        │
                                        │             │
                           NEEDS_MORE──►Supervisor────┘
                           APPROVED
                                │
                                ▼
                        Generate Report ──► END

[RAG Agent 내부 — Agentic RAG]
  Query Transform → Retrieve → Grade Documents
    → NO: Query Rewrite → Retrieve  (max_retry: 5)
    → YES: Draft Generation → Self-Reflection
             → APPROVED: return
             → REVISE: revision (max_revision: 3)
             → RETRIEVE: re-retrieve
```

## Directory Structure

```
battery-strategy-agent/
├── data/
│   ├── lges/           # LGES PDF 문서
│   ├── catl/           # CATL PDF 문서
│   └── market/         # IEA EV Outlook PDF
├── agents/
│   ├── supervisor.py   # 보고서 생성
│   ├── rag_agent.py    # RAG A/B 공통 모듈 (Agentic RAG)
│   ├── search_agent.py # 확증 편향 방지 웹 검색
│   └── critic_agent.py # 균형성 검토
├── graph/
│   └── workflow.py     # LangGraph StateGraph
├── prompts/
│   ├── rag.py          # RAG 추출·반성·수정 프롬프트
│   ├── search.py       # 긍정/부정 쌍 쿼리 프롬프트
│   ├── critic.py       # 균형성 평가 프롬프트
│   └── supervisor.py   # 라우팅·보고서 생성 프롬프트
├── retrieval/
│   ├── ingest.py       # PDF 파싱·청킹·임베딩·FAISS 저장
│   └── retriever.py    # 벡터 검색 (Grade + Rewrite 루프)
├── outputs/            # 생성된 보고서 저장
├── app.py              # 실행 엔트리포인트
├── config.py           # 전역 설정
├── .env                # API 키 설정
└── requirements.txt
```

## Setup & Run

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 3. PDF 파일 배치
cp path/to/lges.pdf   data/lges/
cp path/to/catl.pdf   data/catl/
cp path/to/iea.pdf    data/market/

# 4. 문서 인덱싱 (최초 1회)
python retrieval/ingest.py

# 5. 보고서 생성
python app.py
```

## Contributors

- 본인 이름: Agentic RAG 설계, RAG Agent 개발, 확증 편향 방지 전략 구현, Critic Agent 개발, LangGraph StateGraph 구성
