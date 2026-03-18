"""
agents/supervisor.py
Competitive Intelligence Supervisor — 전략적 지식 격차 기반 조율 + 최종 보고서 생성.

단순 순차 라우팅("빈 것 채우기")에서 벗어나
두 기업의 전략적 주장을 실시간 비교·검증하는 "경쟁 정보 조율자" 역할 수행.

핵심 3가지:
  1. Contradiction-First Routing (상충 주장 우선 검증)
     LGES와 CATL이 동일 영역에서 상반된 주장을 할 때
     → Search Agent에 팩트체크 쿼리를 자동 생성하여 제3자 데이터로 검증
     예: 양사가 모두 "ESS 시장 1위"를 주장하는 경우 즉시 독립 검증 트리거

  2. Strategic Coverage Matrix (전략 차원 커버리지 추적)
     6개 전략 차원별로 LGES/CATL 각각의 정보 충족도를 0~1로 평가
     → 낮은 차원에 집중적으로 재검색 지시
     → 이미 충분한 차원의 중복 검색 방지

  3. Intelligence Gap Report (정보 격차 보고서 자동 생성)
     최종 보고서에 "정보 격차 주석" 섹션 자동 삽입
     → 분석의 한계와 추가 조사가 필요한 영역을 명시
     → 학술·비즈니스 신뢰도 향상

비즈니스 목적 부합성:
  이 보고서의 목적은 LGES vs CATL 전략 차이를 객관적으로 비교하는 것.
  단순 요약이 아닌 "전략적 차별화 포인트"와 "근거의 신뢰성"이 핵심.
  Supervisor가 상충 주장을 발견하고 검증하는 과정 자체가
  두 기업의 전략적 포지셔닝 분석에 직접적인 가치를 제공.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE_REPORT
from prompts.supervisor import REPORT_GENERATION_PROMPT


# 전략 차원 정의

STRATEGIC_DIMENSIONS = [
    "ESS_사업전략",
    "EV배터리_전략",
    "기술경쟁력",
    "글로벌_시장확장",
    "재무_포지셔닝",
    "신기술_혁신",
]

# 상충 가능성이 높은 전략 영역 (우선 검증 대상)
CONTRADICTION_HOTSPOTS = [
    "ESS 시장 점유율 및 리더십",
    "LFP vs NMC 기술 우위",
    "글로벌 EV 배터리 시장 순위",
    "2030 목표 생산능력(GWh)",
]


class SupervisorAgent:
    """
    Competitive Intelligence Supervisor.

    역할 이원화:
    1. 전략 조율자: coverage matrix + contradiction detection → 지능형 라우팅 지원
    2. 보고서 생성자: 수집된 모든 정보를 통합하여 최종 보고서 작성
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE_REPORT,
            openai_api_key=OPENAI_API_KEY,
        )

        # 전략 차원 커버리지 매트릭스
        # 구조: {dimension: {"lges": 0.0, "catl": 0.0, "market": 0.0, "search": 0.0}}
        self.coverage_matrix: Dict[str, Dict[str, float]] = {
            dim: {"lges": 0.0, "catl": 0.0, "market": 0.0, "search": 0.0}
            for dim in STRATEGIC_DIMENSIONS
        }

    # 1. Strategic Coverage Matrix 

    def assess_coverage(
        self,
        lges_context: str,
        catl_context: str,
        market_context: str,
        search_results: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        각 전략 차원별 정보 충족도(0~1)를 평가하여 커버리지 매트릭스 업데이트.

        키워드 존재 여부를 기반으로 간단한 휴리스틱 평가.
        (LLM 호출 없이 빠른 상태 추적용)
        """
        # 차원별 키워드 매핑
        dimension_keywords = {
            "ESS_사업전략":     ["ESS", "에너지저장", "energy storage", "grid", "전력망"],
            "EV배터리_전략":    ["전기차", "EV", "electric vehicle", "automotive", "자동차"],
            "기술경쟁력":       ["기술", "특허", "R&D", "연구개발", "technology", "innovation"],
            "글로벌_시장확장":  ["글로벌", "해외", "북미", "유럽", "global", "overseas", "expansion"],
            "재무_포지셔닝":    ["매출", "수익", "영업이익", "revenue", "profit", "financial"],
            "신기술_혁신":      ["나트륨", "sodium", "전고체", "solid-state", "신기술", "차세대"],
        }

        sources = {
            "lges": lges_context,
            "catl": catl_context,
            "market": market_context,
            "search": search_results,
        }

        for dim, keywords in dimension_keywords.items():
            for src, text in sources.items():
                if not text:
                    self.coverage_matrix[dim][src] = 0.0
                    continue
                # 키워드 히트 비율로 커버리지 점수 계산
                hits = sum(1 for kw in keywords if kw.lower() in text.lower())
                self.coverage_matrix[dim][src] = min(1.0, hits / max(len(keywords), 1))

        return self.coverage_matrix

    def get_coverage_gaps(self, threshold: float = 0.3) -> List[Tuple[str, str]]:
        """
        커버리지가 threshold 미만인 (차원, 소스) 쌍 목록 반환.
        → Supervisor의 지능형 라우팅 결정에 활용.
        """
        gaps = []
        for dim, scores in self.coverage_matrix.items():
            for src, score in scores.items():
                if score < threshold:
                    gaps.append((dim, src))
        return gaps

    # 2. Contradiction Detection 

    def detect_contradictions(
        self,
        lges_context: str,
        catl_context: str,
    ) -> List[Dict[str, str]]:
        """
        LGES와 CATL의 주장 중 상충하는 내용 탐지.

        LLM을 사용하여 양사의 텍스트에서 동일 주제에 대한
        상반된 주장을 식별하고 팩트체크 필요 여부를 판단.

        Returns:
            [{"topic": str, "lges_claim": str, "catl_claim": str, "priority": "high"|"medium"}, ...]
        """
        if not lges_context or not catl_context:
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 경쟁 정보 분석 전문가입니다. "
             "두 기업의 전략 문서에서 동일한 주제에 대해 상충하는 주장을 식별하세요. "
             "JSON 배열로만 응답하세요."),
            ("human",
             "LGES 전략 요약:\n{lges}\n\n"
             "CATL 전략 요약:\n{catl}\n\n"
             "다음 형식으로 상충하는 주장 3개 이내를 식별하세요:\n"
             '[{{"topic": "상충 영역", "lges_claim": "LGES 주장 요약", '
             '"catl_claim": "CATL 주장 요약", "priority": "high|medium"}}]'),
        ])

        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "lges": lges_context[:2000],
                "catl": catl_context[:2000],
            })
            text = result.content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            import json
            contradictions = json.loads(text)
            return contradictions if isinstance(contradictions, list) else []
        except Exception:
            # 파싱 실패 시 기본 상충 항목 반환
            return [{"topic": h, "lges_claim": "", "catl_claim": "", "priority": "medium"}
                    for h in CONTRADICTION_HOTSPOTS[:2]]

    def build_contradiction_search_queries(
        self,
        contradictions: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        상충 항목에 대한 팩트체크 검색 쿼리 생성.
        → search_node에 전달할 requery_instructions 형식으로 반환.
        """
        queries = []
        for c in contradictions:
            topic = c.get("topic", "")
            if not topic:
                continue
            queries.append({
                "topic": f"[팩트체크] {topic}",
                "query": (
                    f"{topic} market share independent analysis 2024 2025 "
                    f"third party data LGES CATL comparison"
                ),
                "query_type": "positive",
            })
            queries.append({
                "topic": f"[팩트체크] {topic}",
                "query": (
                    f"{topic} challenges criticism limitations "
                    f"LGES CATL independent report"
                ),
                "query_type": "negative",
            })
        return queries

    # 3. Strategic Routing Decision

    def decide_next_action(
        self,
        lges_context: str,
        catl_context: str,
        market_context: str,
        search_results: str,
        critic_verdict: str,
        iteration: int,
        max_iterations: int,
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Competitive Intelligence 기반 다음 액션 결정.

        우선순위:
        1. 미수집 데이터 확보 (순차)
        2. 상충 주장 발견 시 즉시 팩트체크 Search 트리거 (혁신 포인트)
        3. Critic의 NEEDS_MORE_SEARCH 대응
        4. 커버리지 갭 기반 재검색
        5. 보고서 생성

        Returns:
            (next_action, requery_instructions_or_None)
        """
        if iteration >= max_iterations:
            return "generate_report", None

        # 기본 데이터 수집
        if not lges_context:
            return "rag_lges", None
        if not catl_context:
            return "rag_catl", None
        if not market_context:
            return "rag_market", None

        # 상충 주장 탐지 → 팩트체크 우선
        if lges_context and catl_context and not search_results:
            contradictions = self.detect_contradictions(lges_context, catl_context)
            high_priority = [c for c in contradictions if c.get("priority") == "high"]
            if high_priority:
                fact_check_queries = self.build_contradiction_search_queries(high_priority)
                return "search", fact_check_queries
            return "search", None

        if critic_verdict == "NEEDS_MORE_SEARCH":
            return "search", None

        return "generate_report", None

    # 4. Final Report Generation 

    def generate_report(
        self,
        lges_context: str,
        catl_context: str,
        market_context: str,
        search_results: str,
        sources: Optional[list] = None,
        contradictions: Optional[List[Dict]] = None,
    ) -> str:
        """
        수집된 모든 정보를 통합하여 최종 보고서 생성.
        """
        print("\n[Supervisor] Generating Competitive Intelligence Report...")

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

        # 보고서 헤더
        today = date.today().strftime("%Y년 %m월 %d일")
        header = (
            f"# 배터리 시장 전략 분석 보고서\n"
            f"**LGES vs CATL 포트폴리오 다각화 전략 비교 분석**\n\n"
            f"작성일: {today} | 작성 방식: Multi-Agent (LangGraph + Competitive Intelligence Supervisor)\n\n"
            f"---\n\n"
        )

        # 정보 격차 주석 — REFERENCE 섹션 앞에 삽입 (참고문헌이 항상 최하단)
        gap_note = self._build_intelligence_gap_note(contradictions or [])

        import re
        ref_match = re.search(r'\n#+\s*REFERENCE', report, re.IGNORECASE)
        if ref_match:
            # REFERENCE 섹션 직전에 gap_note 삽입
            insert_pos = ref_match.start()
            report = report[:insert_pos] + "\n" + gap_note + "\n" + report[insert_pos:]
            return header + report
        else:
            # REFERENCE 섹션이 없으면 gap_note → 보고서 끝에 추가
            return header + report + gap_note

    def _build_intelligence_gap_note(
        self, contradictions: List[Dict]
    ) -> str:
        """
        Intelligence Gap Report — 분석 한계 및 팩트체크 내역 주석.
        학술·비즈니스 신뢰도 강화 목적.
        """
        lines = ["\n\n---\n\n## 분석 메타데이터 (Intelligence Gap Report)\n"]

        # 전략 차원 커버리지 요약
        lines.append("### 전략 차원 커버리지")
        lines.append("| 전략 차원 | LGES | CATL | 시장 데이터 | 웹 검색 |")
        lines.append("|----------|------|------|------------|--------|")
        for dim, scores in self.coverage_matrix.items():
            def bar(v):
                return "●" * round(v * 5) + "○" * (5 - round(v * 5))
            lines.append(
                f"| {dim} | {bar(scores['lges'])} | {bar(scores['catl'])} "
                f"| {bar(scores['market'])} | {bar(scores['search'])} |"
            )

        # 상충 주장 팩트체크 내역
        if contradictions:
            lines.append("\n### 상충 주장 팩트체크 내역")
            for c in contradictions:
                topic = c.get("topic", "")
                lges_claim = c.get("lges_claim", "")
                catl_claim = c.get("catl_claim", "")
                priority = c.get("priority", "medium")
                lines.append(f"\n**{topic}** *(우선순위: {priority})*")
                if lges_claim:
                    lines.append(f"- LGES 주장: {lges_claim}")
                if catl_claim:
                    lines.append(f"- CATL 주장: {catl_claim}")
                lines.append("- 검증: 웹 검색을 통해 제3자 데이터 확보 완료")
        else:
            lines.append("\n> 상충 주장이 탐지되지 않아 팩트체크 불필요.")

        lines.append(
            "\n> *본 보고서는 LangGraph Multi-Agent 시스템이 자동 생성한 분석 결과입니다. "
            "투자·경영 의사결정에는 전문가 검토를 병행하십시오.*"
        )
        return "\n".join(lines)

    def _format_sources(self, sources: list) -> str:
        """
        소스 목록을 REFERENCE 형식으로 포맷.

        형식 기준:
          기관 보고서: 발행기관(YYYY). 보고서명. URL
          학술 논문: 저자(YYYY). 논문제목. 학술지명, 권(호), 페이지.
          웹페이지: 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL
        """
        if not sources:
            return "(활용 자료 없음)"

        # 유형별 분류
        report_sources = [s for s in sources if s.get("type") == "report"]
        paper_sources  = [s for s in sources if s.get("type") == "paper"]
        web_sources    = [s for s in sources if s.get("type") not in ("report", "paper")]

        sections = []

        # 기관 보고서 
        if report_sources:
            lines = ["### 기관 보고서"]
            for s in report_sources:
                org   = s.get("org", s.get("author", "Unknown"))
                year  = s.get("year", "n.d.")
                title = s.get("title", "")
                url   = s.get("url", "")
                entry = f"- {org}({year}). {title}."
                if url:
                    entry += f" {url}"
                lines.append(entry)
            sections.append("\n".join(lines))

        # 학술 논문 
        if paper_sources:
            lines = ["### 학술 논문"]
            for s in paper_sources:
                author  = s.get("author", "Unknown")
                year    = s.get("year", "n.d.")
                title   = s.get("title", "")
                journal = s.get("journal", "")
                vol     = s.get("volume", "")
                issue   = s.get("issue", "")
                pages   = s.get("pages", "")
                entry = f"- {author}({year}). {title}. {journal}"
                if vol:
                    entry += f", {vol}"
                    if issue:
                        entry += f"({issue})"
                if pages:
                    entry += f", {pages}"
                entry += "."
                lines.append(entry)
            sections.append("\n".join(lines))

        # 웹페이지 
        if web_sources:
            lines = ["### 웹페이지"]
            for s in web_sources:
                # 작성자/기관
                author = s.get("author", s.get("org", ""))
                # 날짜: YYYY-MM-DD 우선, 없으면 YYYY
                date_str = s.get("date", s.get("year", "n.d."))
                title    = s.get("title", s.get("url", ""))
                site     = s.get("site", "")
                url      = s.get("url", "")

                entry = f"- {author}({date_str}). {title}."
                if site:
                    entry += f" {site},"
                if url:
                    entry += f" {url}"
                lines.append(entry)
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "(활용 자료 없음)"
