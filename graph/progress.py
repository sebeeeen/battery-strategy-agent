"""
graph/progress.py
ProgressTracker 싱글턴 — 파이프라인 진행 상황을 단순한 단계 형식으로 출력.

출력 형식:
  [Step N/T] 단계명  |  진행률 Z%  (경과시간s 경과)
"""

import time
from typing import Optional

# ─── 단계 레이블 ───────────────────────────────────────────────────────────────
# 최단 경로(루프 없음) 기준 총 단계 수: 6
_TOTAL_STEPS = 6

_STEP_LABELS = {
    "supervisor":       "Supervisor  ─  워크플로우 라우팅 결정",
    "parallel_rag":     "Parallel RAG  ─  LGES · CATL · Market 동시 수집",
    "search":           "Search Agent  ─  웹 검색 (확증 편향 방지 쌍 쿼리)",
    "critic":           "Critic Agent  ─  균형성·편향 독립 평가",
    "human_review":     "Human Review  ─  검토 체크포인트",
    "generate_report":  "Report Generation  ─  최종 보고서 작성",
}


# ─── ProgressTracker ──────────────────────────────────────────────────────────

class ProgressTracker:
    """
    파이프라인 진행 상황 추적 싱글턴.

    사용법:
        from graph.progress import tracker
        tracker.start()               # 파이프라인 시작 시 타이머 초기화
        tracker.update("supervisor")  # 각 노드 진입 시 호출
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._step_counter: int = 0

    def start(self) -> None:
        """파이프라인 시작 — 타이머 및 카운터 초기화."""
        self._start_time = time.time()
        self._step_counter = 0

    def update(self, node: str, detail: str = "") -> None:
        """
        노드 진입 시 진행 상황 출력.

        Args:
            node:   노드 이름 (supervisor, parallel_rag, search, ...)
            detail: 추가 정보 (예: "반복 2/8", "Critic → APPROVED")
        """
        if self._start_time is None:
            self._start_time = time.time()

        self._step_counter += 1
        step = self._step_counter
        pct = min(int(step / _TOTAL_STEPS * 100), 99)
        elapsed = int(time.time() - self._start_time)

        label = _STEP_LABELS.get(node, node)
        suffix = f"  |  {detail}" if detail else ""

        print(
            f"\n[Step {step}/{_TOTAL_STEPS}] {label}{suffix}"
            f"  |  진행률 {pct}%  ({elapsed}s 경과)"
        )

    def done(self) -> None:
        """파이프라인 완료 메시지 출력."""
        if self._start_time is None:
            return
        total = int(time.time() - self._start_time)
        print(f"\n[완료] 파이프라인 종료  |  총 소요시간 {total}s")


# 전역 싱글턴 — 모든 모듈에서 임포트해 사용
tracker = ProgressTracker()
