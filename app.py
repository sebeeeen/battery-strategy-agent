"""
app.py
Battery Strategy Analysis Multi-Agent System — 메인 실행 스크립트.

실행 방법:
  1. PDF 파일 준비:
       data/lges/ 에 LGES PDF 복사
       data/catl/ 에 CATL PDF 복사
       data/market/ 에 IEA EV Outlook PDF 복사

  2. 문서 인덱싱 (최초 1회):
       python retrieval/ingest.py

  3. 보고서 생성:
       python app.py

  4. 결과 확인:
       outputs/agent_report.md  (Markdown)
       outputs/agent_report.pdf (PDF)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import OUTPUTS_DIR, REPORT_FILENAME, REPORT_PDF_FILENAME, DATA_DIR
from graph.workflow import create_app, create_initial_state
from retrieval.ingest import ingest_all


def inject_swot_grid_md(report_md: str) -> str:
    """
    마크다운 보고서에서 SWOT 섹션을 찾아 2x2 시각 그리드 HTML로 변환.

    LLM이 생성한 SWOT 텍스트에서 S/W/O/T 각 항목을 추출해
    색상이 있는 2x2 그리드 HTML 블록으로 교체.
    """
    import re

    def extract_swot_items(text: str, keyword: str) -> str:
        """S/W/O/T 항목 텍스트 추출 — 다음 키워드 전까지."""
        keywords = [r"[*#\s]*W[\s\-–—:(]", r"[*#\s]*O[\s\-–—:(]",
                    r"[*#\s]*T[\s\-–—:(]", r"[*#\s]*S[\s\-–—:(]",
                    r"####", r"###", r"##", r"---"]
        pattern = re.compile(
            rf"(?:^|\n)[*#\s]*{keyword}[\s\-–—:(][^\n]*\n(.*?)(?=(?:\n[*#\s]*(?:{'|'.join(['W','O','T','S'])})[\\s\\-–—:(]|\n###|\n##|\Z)))",
            re.DOTALL | re.IGNORECASE
        )
        m = pattern.search(text)
        if m:
            content = m.group(1).strip()
            # 마크다운 bullet → HTML li
            lines = content.split('\n')
            result = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^[-*•]\s*', '', line)
                if line:
                    result.append(f'<li>{line}</li>')
            return '<ul>' + ''.join(result) + '</ul>' if result else ''
        return ''

    def build_swot_grid(company_label: str, swot_text: str) -> str:
        """SWOT 텍스트 블록 → 2x2 그리드 HTML."""
        s = extract_swot_items(swot_text, 'S')
        w = extract_swot_items(swot_text, 'W')
        o = extract_swot_items(swot_text, 'O')
        t = extract_swot_items(swot_text, 'T')

        # fallback: 내용이 없으면 원문 일부 표시
        if not any([s, w, o, t]):
            return swot_text

        return f"""
<div class="swot-wrapper">
<div class="swot-label-row"><span>💪 내부 요인</span><span></span></div>
<div class="swot-label-row"><span>강점 (Strengths)</span><span>약점 (Weaknesses)</span></div>
<div class="swot-grid">
  <div class="swot-cell swot-s">
    <span class="swot-cell-title">S — 강점 (내부)</span>
    {s or '<p>-</p>'}
  </div>
  <div class="swot-cell swot-w">
    <span class="swot-cell-title">W — 약점 (내부)</span>
    {w or '<p>-</p>'}
  </div>
  <div class="swot-cell swot-o">
    <span class="swot-cell-title">O — 기회 (외부)</span>
    {o or '<p>-</p>'}
  </div>
  <div class="swot-cell swot-t">
    <span class="swot-cell-title">T — 위협 (외부)</span>
    {t or '<p>-</p>'}
  </div>
</div>
<div class="swot-label-row"><span>🌍 외부 요인</span><span></span></div>
</div>
"""

    # SWOT 섹션 찾기 — "LGES SWOT" 또는 "CATL SWOT" 블록 교체
    swot_pattern = re.compile(
        r'(#{2,4}[^\n]*SWOT[^\n]*\n)(.*?)(?=\n#{1,4}\s|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    def replace_swot(m):
        header = m.group(1)
        body = m.group(2)
        company = 'LGES' if 'LGES' in header or 'LG' in header else 'CATL'
        grid_html = build_swot_grid(company, body)
        # 그리드가 실제로 생성됐으면 교체, 아니면 원문 유지
        if '<div class="swot-grid">' in grid_html:
            return header + '\n' + grid_html + '\n'
        return m.group(0)

    return swot_pattern.sub(replace_swot, report_md)


def check_env() -> bool:
    """필수 환경 변수 및 파일 확인."""
    ok = True

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=sk-... 를 추가해주세요.")
        ok = False

    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY 가 없습니다. 웹 검색이 비활성화됩니다.")

    return ok


def prepare_data_folders(uploads_dir: str = None):
    """
    uploads 폴더에서 data/ 하위 폴더로 PDF 자동 복사.
    직접 실행 시 편의 기능: data/ 에 이미 파일이 있으면 스킵.
    """
    if uploads_dir is None:
        return

    uploads = Path(uploads_dir)
    if not uploads.exists():
        return

    copy_rules = [
        ("lges", ["LG_Energy", "LGES", "lges"]),
        ("catl", ["CATL", "catl", "ESG"]),  # CATL ESG Report
        ("market", ["GlobalEV", "IEA", "EV_Outlook"]),
    ]

    for doc_type, keywords in copy_rules:
        target_dir = DATA_DIR / doc_type
        # 이미 파일 있으면 스킵
        if list(target_dir.glob("*.pdf")):
            continue
        for pdf in uploads.glob("*.pdf"):
            if any(kw in pdf.name for kw in keywords):
                dest = target_dir / pdf.name
                shutil.copy2(pdf, dest)
                print(f"  Copied: {pdf.name} → data/{doc_type}/")
                break


def save_report(report_md: str) -> tuple[str, str]:
    """보고서를 Markdown 및 PDF로 저장."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Markdown 저장
    md_path = OUTPUTS_DIR / REPORT_FILENAME
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n✅ Markdown saved: {md_path}")

    # PDF 변환 (weasyprint)
    pdf_path = OUTPUTS_DIR / REPORT_PDF_FILENAME
    try:
        import markdown
        from weasyprint import HTML, CSS

        # SWOT 섹션을 시각적 2x2 그리드로 변환 (마크다운 → HTML 전 처리)
        processed_md = inject_swot_grid_md(report_md)

        html_content = markdown.markdown(
            processed_md,
            extensions=["tables", "fenced_code", "toc"],
        )

        # HTML 래퍼 (스타일 포함)
        full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  body {{
    font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', 'Nanum Gothic', sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1a1a1a;
  }}
  h1 {{ font-size: 18pt; color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 14pt; color: #2c5f8a; margin-top: 22px; margin-bottom: 8px; }}
  h3 {{ font-size: 12pt; color: #3a7ab5; margin-top: 16px; margin-bottom: 6px; }}
  h4 {{ font-size: 11pt; color: #444; margin-top: 12px; margin-bottom: 4px; }}
  p {{ margin: 6px 0; }}
  ul, ol {{ margin: 6px 0; padding-left: 20px; }}
  li {{ margin: 3px 0; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9.5pt;
  }}
  th {{
    background-color: #1a3a5c;
    color: white;
    padding: 7px 10px;
    text-align: left;
  }}
  td {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
  tr:nth-child(even) {{ background-color: #f5f8fc; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 9pt; }}
  blockquote {{
    border-left: 4px solid #2c5f8a;
    margin: 8px 0;
    padding-left: 14px;
    color: #555;
  }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 16px 0; }}

  /* ── SWOT 2x2 그리드 ── */
  .swot-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 0;
    width: 100%;
    margin: 14px 0 20px 0;
    border: 2px solid #ccc;
    border-radius: 6px;
    overflow: hidden;
    font-size: 9.5pt;
  }}
  .swot-cell {{
    padding: 12px 14px;
    vertical-align: top;
  }}
  .swot-cell-title {{
    font-size: 11pt;
    font-weight: bold;
    margin-bottom: 8px;
    display: block;
  }}
  .swot-s {{ background-color: #e8f5e9; border-right: 1px solid #ccc; border-bottom: 1px solid #ccc; }}
  .swot-s .swot-cell-title {{ color: #2e7d32; }}
  .swot-w {{ background-color: #fff3e0; border-bottom: 1px solid #ccc; }}
  .swot-w .swot-cell-title {{ color: #e65100; }}
  .swot-o {{ background-color: #e3f2fd; border-right: 1px solid #ccc; }}
  .swot-o .swot-cell-title {{ color: #1565c0; }}
  .swot-t {{ background-color: #fce4ec; }}
  .swot-t .swot-cell-title {{ color: #b71c1c; }}
  .swot-axis-label {{
    text-align: center;
    font-size: 8pt;
    color: #888;
    margin: 2px 0 6px 0;
  }}
  .swot-wrapper {{
    margin: 10px 0 20px 0;
  }}
  .swot-label-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    font-size: 8pt;
    color: #666;
    text-align: center;
    margin-bottom: 2px;
    font-style: italic;
  }}

  @page {{
    margin: 1.5cm 1.8cm;
    @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 8.5pt; color: #888; }}
  }}
</style>
</head>
<body>
{html_content}
</body>
</html>"""

        HTML(string=full_html).write_pdf(str(pdf_path))
        print(f"✅ PDF saved: {pdf_path}")

    except ImportError:
        print("⚠️  weasyprint/markdown 미설치. PDF 변환 스킵.")
        print(f"   pip install weasyprint markdown 후 재실행")
    except Exception as e:
        print(f"⚠️  PDF 변환 오류: {e}")

    return str(md_path), str(pdf_path)


def run_pipeline(
    uploads_dir: str = None,
    force_rebuild: bool = False,
    max_iterations: int = 8,
):
    """
    전체 Multi-Agent 파이프라인 실행.

    Args:
        uploads_dir: PDF 업로드 폴더 경로 (자동 복사 사용 시)
        force_rebuild: FAISS 인덱스 재구축 여부
        max_iterations: Supervisor 최대 반복 횟수
    """
    start_time = time.time()

    print("=" * 65)
    print("  Battery Strategy Analysis — Multi-Agent System")
    print("  LG Energy Solution vs CATL Portfolio Strategy Report")
    print("=" * 65)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 환경 확인
    if not check_env():
        sys.exit(1)

    # 2. PDF 자동 복사 (uploads_dir 지정 시)
    if uploads_dir:
        print("[Step 1] Auto-copying PDF files...")
        prepare_data_folders(uploads_dir)

    # 3. 문서 인덱싱
    print("\n[Step 2] Building FAISS indexes...")
    try:
        ingest_all(force_rebuild=force_rebuild)
    except FileNotFoundError as e:
        print(f"\n❌ PDF 파일을 찾을 수 없습니다:\n  {e}")
        print("\n수동으로 PDF 파일을 아래 폴더에 복사해주세요:")
        print(f"  data/lges/   ← LGES PDF")
        print(f"  data/catl/   ← CATL PDF")
        print(f"  data/market/ ← IEA EV Outlook PDF")
        sys.exit(1)

    # 4. Multi-Agent 파이프라인 실행
    print("\n[Step 3] Running Multi-Agent Analysis Pipeline...")
    print("-" * 65)

    app = create_app()
    initial_state = create_initial_state(max_iterations=max_iterations)

    final_state = app.invoke(initial_state)

    # 5. 보고서 저장
    report = final_state.get("final_report", "")
    if not report:
        print("❌ 보고서 생성에 실패했습니다.")
        sys.exit(1)

    print("\n[Step 4] Saving report...")
    md_path, pdf_path = save_report(report)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  ✅ Pipeline completed in {elapsed:.1f}s")
    print(f"  📄 Report (MD):  {md_path}")
    print(f"  📄 Report (PDF): {pdf_path}")
    print("=" * 65)

    return final_state


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Battery Strategy Analysis Multi-Agent System"
    )
    parser.add_argument(
        "--uploads",
        type=str,
        default=None,
        help="PDF 업로드 폴더 경로 (자동 복사 사용 시)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="FAISS 인덱스 강제 재구축",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=8,
        help="Supervisor 최대 반복 횟수 (기본: 8)",
    )
    args = parser.parse_args()

    run_pipeline(
        uploads_dir=args.uploads,
        force_rebuild=args.rebuild,
        max_iterations=args.max_iter,
    )
