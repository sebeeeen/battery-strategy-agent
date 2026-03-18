# 1. 워크플로우 설계

## 1-1.  목표 (Goal)

> LGES와 CATL의 포트폴리오 다각화 전략을 객관적으로 비교·분석하는 보고서를 Multi-Agent 시스템으로 자동 생성한다.
두 기업의 연간 보고서(PDF) + 실시간 웹 데이터를 통합하여 전략적 차별화 포인트와 시장 포지셔닝 차이를 규명하는 것이 핵심이다.
> 

## 1-2. 설계 기준 (Criteria)

| 기준 | 내용 |
| --- | --- |
| **데이터 격리** | LGES/CATL 문서가 서로 혼합되지 않도록 독립 에이전트 + 독립 FAISS 인덱스 운용 |
| **확증 편향 방지** | 긍정·부정 쌍 쿼리(Dual-Query) 전략으로 일방적 정보 수집 차단 |
| **자기 교정** | Self-Reflection 루프로 저품질 드래프트 자동 수정 (max_revision: 3) |
| **독립 검증** | Critic Agent가 양사 정보의 균형성을 독립적으로 평가 |
| **재현 가능성** | 환경변수 기반 설정으로 다른 기업 쌍에도 즉시 적용 가능 |
| **병렬 처리** | 독립 RAG 에이전트 3개를 `asyncio.gather()`로 동시 실행하여 처리 시간 단축 |

## 1-3. 과제 (Task)

1. **병렬 처리:** LGES + CATL + Market RAG를 `asyncio.gather()`로 동시 실행 (Parallel RAG Node)
2. **웹 검색**: 최신 시장 데이터 수집 + 상충 주장 팩트체크 (Search Agent)
3. 정보 균형성 평가 및 편향 감지 (Critic Agent)
4. 비교 분석 보고서 자동 생성 (Supervisor)

## 1-4. 제어 전략 (Control Strategy)

| 파라미터 | 값 | 동작 |
| --- | --- | --- |
| `max_iterations` | 8 | 초과 시 강제 종료 |
| `max_retry` | 5 | 쿼리 재작성 후 fallback |
| `max_revision` | 3 | 최선 결과 반환 |
| `balance_score` | < 0.6 | `NEEDS_MORE_SEARCH` 트리거 |

---

# 2. 에이전트 정의

## 2-1. 에이전트 구성표

| 에이전트 | 파일 | 역할 | 도구 | 실행 방식 |
| --- | --- | --- | --- | --- |
| **LGESRagAgent** | `agents/lges_rag_agent.py` | LGES 문서 전용 Agentic RAG | `retrieve_lges_documents` @tool | **병렬**  |
| **CATLRagAgent** | `agents/catl_rag_agent.py` | CATL 문서 전용 Agentic RAG | `retrieve_catl_documents` @tool | **병렬**  |
| **RAGAgent(market)** | `agents/rag_agent.py` | 시장 데이터 RAG | `retrieve_market_documents` @tool | **병렬**  |
| **SearchAgent** | `agents/search_agent.py` | 확증 편향 방지 웹 검색 | `search_positive_perspective`, `search_critical_perspective`, `resolve_strategic_contradiction` @tool | 순차 |
| **CriticAgent** | `agents/critic_agent.py` | 균형성 독립 평가 | LLM 직접 호출 | 순차 |
| **SupervisorAgent** | `agents/supervisor.py` | 조율 + 보고서 생성 | Contradiction Detection, Coverage Matrix | 순차 |

## 2-2. LGESRagAgent / CATLRagAgent 독립 클래스 분리

- **독립 FAISS 인덱스**
    
    `LGESRagAgent`는 `faiss_lges` 인덱스만, `CATLRagAgent`는 `faiss_catl` 인덱스만 접근 → 물리적 데이터 격리
    
- **전용 @tool 바인딩**
    
    `retrieve_lges_documents`와 `retrieve_catl_documents`는 별도의 `@tool` 함수 → LLM이 잘못된 인덱스를 호출하는 가능성 원천 차단
    
- **독립 LLM 인스턴스**
    
    각 에이전트가 별도 `ChatOpenAI()` 인스턴스 소유 → 대화 히스토리 혼합 없음
    
- **특화 토픽 목록**
    
    LGES는 `EV 전략·ESS·기술역량` 중심, CATL은 `나트륨이온·LFP·글로벌 확장` 중심으로 분리
    

## 2-3. @tool 데코레이터 적용

`langchain_core.tools.@tool` 데코레이터를 통해 각 검색 함수를 LLM이 직접 호출 가능한 Tool로 등록한다.

1. **JSON Schema 자동 생성**
    
    `llm.bind_tools()`로 OpenAI Function Calling API 활성화, 별도 스키마 작성 불필요
    
2. **LLM 자율 검색 결정**
    
    LLM이 "언제 검색이 필요한지" 스스로 판단 (진정한 Agentic 행동)
    
3. **입력 유효성 검사**
    
    Pydantic 기반 타입 검증 내장으로 잘못된 쿼리 차단
    
4. **LangSmith 추적**
    
    도구 호출 단계가 시각적으로 분리되어 디버깅 용이
    
5. **비동기 지원**
    
    `arun()`으로 parallel_rag_node 내 병렬 다중 검색 가능
    

---

# 3. Agentic RAG 설계 (CRAG + Self-Reflection)

## 3-1. 7단계 파이프라인

| 단계 | 이름 | 설명 |
| --- | --- | --- |
| Step 1 | Query Transformation | 회사명 + 주제 결합 |
| Step 2 | Retrieve | FAISS `similarity_search` |
| Step 3 | Grade Documents | LLM: YES/NO 관련성 판단 |
| Step 4 | Query Rewrite | 미흡 시 쿼리 재작성 (↔ Step 2 반복, max_retry: 5) |
| Step 5 | Draft Generation | `RAG_EXTRACTION_PROMPT` |
| Step 6 | Self-Reflection | APPROVED / REVISE / RETRIEVE (↔ Step 5/2 반복, max_revision: 3) |
| Step 7 | Memory Update | State 반환 |

## 3-2. Self-Reflection 판정 기준

| Verdict | 조건 | 다음 액션 |
| --- | --- | --- |
| `APPROVED` | 구체적 수치·데이터 포함, 분석 깊이 충분 | 결과 반환 |
| `REVISE` | 데이터는 있으나 분석 관점·구체성 부족 | Draft 수정 (guidance 전달) |
| `RETRIEVE` | 핵심 정보 누락, 컨텍스트 불충분 | 추가 쿼리로 재검색 |

## 3-3. CRAG 파라미터

```python
CHUNK_SIZE      = 1200   # 청크 크기 (토큰 단위 근사)
CHUNK_OVERLAP   = 200    # 청크 겹침 (문맥 연속성 보장)
TOP_K           = 6      # 검색 문서 수
MAX_RETRY       = 5      # 최대 쿼리 재작성 횟수
MAX_REVISION    = 3      # 최대 드래프트 수정 횟수
SCORE_THRESHOLD = 0.3    # FAISS 유사도 임계값
```

---

# 4. 임베딩 모델 선정

## 4-1. 후보 모델 비교

| 모델 | 언어 지원 | 차원 | 특징 | 비용 |
| --- | --- | --- | --- | --- |
| `ko-sroberta-multitask` | 한국어 특화 | 768 | 한국어 최적화 | 무료 |
| `multilingual-e5-large` | 다국어 | 1024 | Microsoft, 높은 성능 | 무료 |
| **`BAAI/bge-m3`**  | **100+ 언어** | **1024** | **Dense+Sparse+ColBERT** | **무료** |

## 4-2. BAAI/bge-m3 선정 이유

> 분석 문서가 영문(LGES 연간보고서, IEA 보고서) + 한글(분석 쿼리) 혼재 환경이므로
단일 모델로 양쪽 언어를 고품질로 처리할 수 있는 모델이 필요하다.
> 
1. **다국어 완전 지원 (100+ 언어)**
    
    영문 PDF 청크와 한글 쿼리 간 Cross-lingual Retrieval 최적
    
2. **3중 표현 방식**
    
    Dense(의미), Sparse(키워드), ColBERT(세밀 매칭) 동시 지원 → 배터리 전문 용어 처리에 유리
    
3. **MTEB 벤치마크 최상위**
    
    다국어 검색 태스크에서 오픈소스 중 최고 수준
    
4. **정규화 임베딩**
    
    `normalize_embeddings=True`로 코사인 유사도 최적화
    

---

# 5. Supervisor 설계: Competitive Intelligence Supervisor

> **설계 철학:** 단순 "빈 것 채우기" 라우터에서 벗어나, 비즈니스 목적(두 기업의 전략 차이 규명)에 직접 기여하는 **경쟁 정보 조율자**로 재설계
> 

## 5-1. Contradiction-First Routing (상충 주장 우선 검증)

LGES RAG + CATL RAG 완료 후 `SupervisorAgent.detect_contradictions()`가 자동 실행되며, 상충 발견 시 팩트체크 쿼리를 자동 생성하여 Search Agent를 즉시 호출한다.

**실제 적용 예시:**

1. LGES: "ESS 시장 성장 주도" VS CATL: "글로벌 ESS 1위 유지"
2. Supervisor가 상충 탐지
3. `resolve_strategic_contradiction(@tool)` 자동 호출

## 5-2. Strategic Coverage Matrix (전략 차원 커버리지 추적)

```python
STRATEGIC_DIMENSIONS = [
    "ESS_사업전략", "EV배터리_전략", "기술경쟁력",
    "글로벌_시장확장", "재무_포지셔닝", "신기술_혁신"
]

# 각 차원별 0~1 충족도 추적
coverage_matrix = {
    "ESS_사업전략": {"lges": 0.8, "catl": 0.9, "market": 0.6, "search": 0.7},
    "신기술_혁신":  {"lges": 0.4, "catl": 0.9, "market": 0.3, "search": 0.5},
    ...
}
```

낮은 차원을 집중 보완하고, 이미 충분한 차원의 중복 검색을 방지한다.

## 5-3. Intelligence Gap Report (정보 격차 보고서 자동 삽입)

최종 보고서에 전략 차원별 커버리지 매트릭스와 상충 주장 팩트체크 내역이 자동으로 삽입된다. 

예시:

| 전략 차원 | LGES | CATL | 시장 데이터 | 웹 검색 |
| --- | --- | --- | --- | --- |
| ESS_사업전략 | ●●●●● | ●●●●● | ●●●○○ | ●●●●○ |
| 신기술_혁신 | ●●○○○ | ●●●●● | ●○○○○ | ●●●○○ |

---

# 6. Human-in-the-Loop (HiTL) 설계

## 6-1. 적용 **이유**

- 배터리 전략 보고서는 투자·경영 의사결정에 활용될 수 있으므로 데이터 품질 검증이 중요
- Critic Agent가 "APPROVED"를 내렸더라도 실제 전문가 검토가 필요한 영역이 있을 수 있음
- 사람이 `coverage_matrix`와 `critic_feedback`을 검토 후 추가 검색 주제를 지정할 수 있음

## 6-2. HiTL 실행 흐름

1. `interrupt_before=["generate_report"]`로 보고서 생성 직전 자동 중단 
2. 사람이 `critic_feedback`과 `coverage_matrix`를 확인 후 승인 또는 추가 지시 
3. `app.stream(None, config)`으로 재개

---

# 7. Critic Agent 판단 기준

## 7-1. 균형성 평가 항목

Critic Agent는 수집된 정보의 균형성과 완결성을 다음 4개 항목으로 평가하여 `balance_score`(0~1)를 산출한다.

| 평가 항목 | 가중치 | 평가 기준 |
| --- | --- | --- |
| **정보 균형성 (Balance)** | 40% | LGES와 CATL 정보량·심도가 대등하게 수집되었는가 |
| **커버리지 충족도 (Coverage)** | 30% | 6개 전략 차원(ESS, EV, 기술경쟁력, 글로벌확장, 재무, 신기술)이 충분히 다뤄졌는가 |
| **상충 검증 완료 (Contradiction)** | 20% | 탐지된 상충 주장이 웹 검색을 통해 팩트체크되었는가 |
| **소스 다양성 (Source Diversity)** | 10% | RAG 문서 + 웹 검색 결과가 다양하게 포함되었는가 |
| **합계** | **100%** |  |

## 7-2. 점수 산정 방식

각 항목은 **1~5점 척도**로 평가하며, 가중치를 적용하여 `balance_score`(0~1)를 계산한다.

| 점수 | 의미 |
| --- | --- |
| 5 | 매우 충족 |
| 4 | 충족 |
| 3 | 보통 |
| 2 | 미흡 |
| 1 | 매우 미흡 |

> [정보 균형성 4점 × 40% = 1.6점 / 커버리지 충족도 3점 × 30% = 0.9점
→ 4개 항목 가중합 후 5점 만점 대비 정규화 → `balance_score`
> 

## 7-3. 최종 판단 기준

| balance_score | 판단 | 다음 액션 |
| --- | --- | --- |
| **0.8 이상** | APPROVED | 보고서 생성 단계로 진행 |
| **0.6 ~ 0.79** | APPROVED (주의) | 부족 항목을 `critic_feedback`에 명시 후 진행 |
| **0.6 미만** | NEEDS_MORE_SEARCH | Supervisor에게 재검색 지시 (`requery_instructions` 생성) |

## 7-4. 에이전트 연계

| Critic 평가 항목 | 관련 에이전트 |
| --- | --- |
| 정보 균형성 | LGESRagAgent, CATLRagAgent |
| 커버리지 충족도 | 전체 (coverage_matrix 기반) |
| 상충 검증 완료 | SearchAgent (`resolve_strategic_contradiction`) |
| 소스 다양성 | Market RAGAgent, SearchAgent |

---

# 8. State & Graph Flow

## 8-1. BatteryAnalysisState 전체 필드

```python
class BatteryAnalysisState(TypedDict):
    # 수집 정보
    lges_context: str          # LGESRagAgent 결과
    catl_context: str          # CATLRagAgent 결과
    market_context: str        # Market RAGAgent 결과
    search_results: str        # 웹 검색 결과 (텍스트)
    search_sources: List[Dict] # 검색 소스 메타데이터

    # Competitive Intelligence
    contradictions: List[Dict] # 탐지된 상충 주장 목록
    coverage_matrix: Dict      # 6차원 × 4소스 커버리지 매트릭스

    # Critic 피드백
    critic_feedback: str       # 평가 요약
    critic_verdict: str        # "APPROVED" | "NEEDS_MORE_SEARCH"
    requery_instructions: List[Dict]

    # Human-in-the-Loop
    human_approved: bool       # 사람의 최종 승인
    human_notes: str           # 검토 메모 (추가 지시사항)

    # 제어
    iteration: int             # 현재 반복 횟수
    max_iterations: int        # 최대 반복 (기본: 8)
    next_action: str           # 다음 액션 레이블

    # 출력
    final_report: str          # 최종 Markdown 보고서
```

---

# 9. 아키텍처 다이어그램

## 9-1. 전체 워크플로우

![전체 워크플로우](./assets/전체%20워크플로우.png)

## 9-2. 병렬 실행 시퀀스 다이어그램

![병렬 실행 시퀀스](./assets/병렬%20실행%20시퀀스%20다이어그램.png)

## 9-3. Agentic RAG 내부 플로우

![Agentic RAG 플로우](./assets/Agentic%20RAG%20내부%20플로우.png)

---

# 10. 100페이지 문서 구성

| 문서 | 파일명 | 사용 페이지 | 페이지 수 | 선정 기준 |
| --- | --- | --- | --- | --- |
| **LGES 연간보고서** | `lges_report.pdf` | p.6 ~ p.30 | 25p | 사업 전략·포트폴리오·기술 역량 섹션 |
| **CATL 연간보고서** | `catl_report.pdf` | p.1 ~ p.45 | 45p | 회사 개요·전략·기술·시장 전체 |
| **IEA EV Outlook** | `iea_ev_outlook.pdf` | p.11 ~ p.40 | 30p | Executive Summary + EV 시장 동향 |
| **합계** | — | — | **100p** | — |

---

# 11. 최종 보고서 목차 구성

Supervisor Agent가 생성하는 최종 투자 분석 보고서의 목차 구성이다.

| 챕터 | 제목 | 포함 항목 | 작성 목적 |
| --- | --- | --- | --- |
| **1. 종합 요약** | Executive Summary | 분석 대상(LGES·CATL), 핵심 전략 차이, 주요 강점·리스크, 포트폴리오 다각화 방향 제언 | 보고서 전체 결론을 먼저 제시 |
| **2. 시장 배경** | Market Context | 글로벌 EV 시장 성장률, ESS 수요 확대, 배터리 원자재 가격 동향, 정책 환경 | 개별 기업 전략을 산업 맥락 안에서 평가하기 위한 배경 제공 |
| **3. LGES 전략 분석** | LGES 포트폴리오 다각화 | ESS 사업 전략, EV 배터리 포지셔닝, 기술 경쟁력, 글로벌 생산 거점, 주요 고객사 | LGES의 전략적 방향성과 강약점 규명 |
| **4. CATL 전략 분석** | CATL 포트폴리오 다각화 | 나트륨이온·LFP 기술 전략, 글로벌 확장 계획, 수직 계열화, 주요 고객사 및 파트너십 | CATL의 전략적 방향성과 강약점 규명 |
| **5. 전략 비교 분석** | 핵심 차별화 포인트 | 6개 전략 차원별 비교표, 기술 격차, 시장 포지셔닝 차이, 상충 주장 팩트체크 결과 | 두 기업의 전략적 차이를 객관적으로 규명 |
| **6. 경쟁 역학 분석** | 시장 내 경쟁 구도 | 시장 점유율 추이, 기술 투자 비교, 가격 경쟁력, 공급망 전략 차이 | 경쟁 격화 속 각사의 생존 전략 평가 |
| **7. 리스크 분석** | 양사 리스크 비교 | 기술 리스크, 공급망 리스크, 지정학적 리스크, 재무 리스크, 규제 리스크 | 포트폴리오 투자 전 핵심 위험 요소 정리 |
| **8. 포트폴리오 시사점** | 투자 전략 제언 | 양사 대비 투자 매력도 비교, 포트폴리오 배분 전략, 단기·중장기 전망 | 포트폴리오 다각화 의사결정 지원 |
| **9. 분석 메타데이터** | Intelligence Gap Report | 전략 차원별 커버리지 매트릭스, 상충 주장 팩트체크 내역, Critic Agent 평가 결과 | 분석 근거의 신뢰성 제시 |
| **10. 참고문헌** | References | 기관보고서(연간보고서·IEA), 학술논문, 웹페이지(뉴스·검색 결과) | 분석 근거 투명하게 제시 |

## 참고문헌 유형 분류

| 유형 | 형식 예시 |
| --- | --- |
| **기관보고서** | LG에너지솔루션. (2023). *Annual Report 2023*. LG Energy Solution. |
| **학술논문** | Smith, J., & Lee, K. (2024). Battery market dynamics. *Journal of Energy Storage*, 45, 123-135. |
| **웹페이지** | IEA. (2024). *Global EV Outlook 2024*. https://www.iea.org/reports/global-ev-outlook-2024 |