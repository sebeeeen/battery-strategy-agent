from langchain_core.prompts import ChatPromptTemplate

SUPERVISOR_ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a supervisor agent coordinating a battery industry analysis pipeline. "
     "Based on the current state, decide what action to take next.\n\n"
     "Available actions:\n"
     "- 'rag_lges': Retrieve LGES strategy information from PDF documents\n"
     "- 'rag_catl': Retrieve CATL strategy information from PDF documents\n"
     "- 'rag_market': Retrieve market background information from IEA report\n"
     "- 'search': Run web searches for latest information\n"
     "- 'critic': Have the critic evaluate balance and completeness\n"
     "- 'generate_report': Generate the final report (only when all info is gathered)\n"
     "- 'end': Terminate\n\n"
     "Decision criteria:\n"
     "1. If LGES context is empty → 'rag_lges'\n"
     "2. If CATL context is empty → 'rag_catl'\n"
     "3. If market context is empty → 'rag_market'\n"
     "4. If search results are empty → 'search'\n"
     "5. If critic has not reviewed yet → 'critic'\n"
     "6. If critic says NEEDS_MORE_SEARCH → 'search'\n"
     "7. If all info gathered and critic approved → 'generate_report'\n\n"
     "Return ONLY the action string, nothing else."),
    ("human",
     "Current State:\n"
     "- lges_context: {lges_context_status}\n"
     "- catl_context: {catl_context_status}\n"
     "- market_context: {market_context_status}\n"
     "- search_results: {search_results_status}\n"
     "- critic_feedback: {critic_feedback}\n"
     "- iteration: {iteration}\n"
     "- max_iterations: {max_iterations}\n\n"
     "What action should be taken next?"),
])


# Report Generation Prompt

REPORT_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior strategy analyst writing a comprehensive battery industry report in Korean. "
     "Generate a professional, data-driven report based on the provided information.\n\n"
     "REPORT STRUCTURE (mandatory):\n"
     "# SUMMARY\n"
     "(전체 보고서 핵심 요약, 1/2 페이지 이내)\n\n"
     "# 2. 시장 배경 — 글로벌 배터리 시장 환경 변화\n\n"
     "# 3. LG에너지솔루션 포트폴리오 다각화 전략\n"
     "## 3.1 핵심 사업 방향\n"
     "## 3.2 주요 경쟁력\n\n"
     "# 4. CATL 포트폴리오 다각화 전략\n"
     "## 4.1 핵심 사업 방향\n"
     "## 4.2 주요 경쟁력\n\n"
     "# 5. 전략 비교 분석\n"
     "## 5.1 핵심 전략 비교 매트릭스\n"
     "(표 형식으로 작성)\n"
     "## 5.2 SWOT 분석\n"
     "(LGES SWOT — 내부 S/W + 외부 O/T)\n"
     "(CATL SWOT — 내부 S/W + 외부 O/T)\n\n"
     "# 6. 종합 시사점\n\n"
     "# REFERENCE\n"
     "(실제 활용한 자료 목록만 기재, 형식 준수)\n\n"
     "IMPORTANT RULES:\n"
     "- Every claim must be grounded in the provided sources\n"
     "- Include specific numbers, percentages, and dates where available\n"
     "- SWOT must clearly separate internal (S/W) from external (O/T)\n"
     "- SUMMARY must be concise (max 300 words)\n"
     "- Write entirely in Korean (except proper nouns and technical terms)\n"
     "- REFERENCE section rules:\n"
     "  * 보고서 작성에 실제로 활용한 자료 목록만 기재 (미사용 자료 제외)\n"
     "  * 자료 유형별로 구분하여 작성 (해당 유형이 없으면 해당 소제목 생략)\n"
     "  * 형식:\n"
     "    기관 보고서: 발행기관(YYYY). 보고서명. URL\n"
     "    학술 논문: 저자(YYYY). 논문제목. 학술지명, 권(호), 페이지.\n"
     "    웹페이지: 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL\n"
     "  * 예시:\n"
     "    ### 기관 보고서\n"
     "    - LG에너지솔루션(2024). LG Energy Solution Annual Report 2023. https://...\n"
     "    - IEA(2024). Global EV Outlook 2024. https://www.iea.org/...\n"
     "    ### 웹페이지\n"
     "    - SNE Research(2024-03-15). 글로벌 배터리 시장 점유율 2024. SNE Research, https://...\n"
     "  * URL이 없는 자료는 URL 항목 생략 가능\n"
     "  * 검색 결과 source_list에 있는 실제 URL만 사용할 것"),
    ("human",
     "=== LGES Strategy Information ===\n{lges_context}\n\n"
     "=== CATL Strategy Information ===\n{catl_context}\n\n"
     "=== Market Background ===\n{market_context}\n\n"
     "=== Latest Web Search Information ===\n{search_results}\n\n"
     "=== Source Documents ===\n{source_list}\n\n"
     "Please write the complete battery strategy analysis report in Korean:"),
])
