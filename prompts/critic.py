"""
prompts/critic.py
Critic Agent 프롬프트 — 수집 정보의 균형성 검토 및 편향 탐지.

역할:
- RAG + 웹 검색으로 수집된 모든 정보를 독립적으로 검토
- 정보의 균형성 평가 (긍/부정 균형, LGES/CATL 균형)
- 편향 발견 시 재검색 지시 (어떤 주제/방향으로 재검색할지 명시)
"""

from langchain_core.prompts import ChatPromptTemplate

# ─── Balance Evaluation Prompt ────────────────────────────────────────────────

CRITIC_BALANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an independent fact-checker and bias detector for a battery industry report. "
     "Evaluate the collected information for:\n\n"
     "1. BALANCE CHECK — Are both positive and negative aspects covered for each company?\n"
     "2. COMPANY PARITY — Is the depth of analysis roughly equal for LGES and CATL?\n"
     "3. DATA QUALITY — Are claims supported by specific numbers/data?\n"
     "4. COVERAGE CHECK — Are all required topics covered?\n"
     "   Required topics: market background, LGES strategy, CATL strategy, comparative analysis\n\n"
     "Return a JSON evaluation:\n"
     "{{\n"
     '  "is_balanced": true | false,\n'
     '  "balance_score": 0.0-1.0,\n'
     '  "issues": ["issue1", "issue2", ...],\n'
     '  "missing_topics": ["topic1", ...],\n'
     '  "requery_instructions": [\n'
     '    {{"topic": "...", "direction": "positive|negative|factual", "reason": "..."}}\n'
     "  ],\n"
     '  "verdict": "APPROVED" | "NEEDS_MORE_SEARCH"\n'
     "}}"),
    ("human",
     "Collected Information Summary:\n\n"
     "=== LGES RAG Context ===\n{lges_context}\n\n"
     "=== CATL RAG Context ===\n{catl_context}\n\n"
     "=== Market Context ===\n{market_context}\n\n"
     "=== Web Search Results ===\n{search_results}\n\n"
     "Please evaluate the balance and completeness of the collected information:"),
])


# ─── Re-query Direction Prompt ────────────────────────────────────────────────

CRITIC_REQUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are providing targeted search guidance to fix identified information gaps or biases. "
     "Generate specific, actionable search queries to address the issues found."),
    ("human",
     "Issues identified:\n{issues}\n\n"
     "Missing topics:\n{missing_topics}\n\n"
     "Generate specific search queries to fill these gaps. "
     'Return as JSON list: [{{"topic": "...", "query": "...", "query_type": "positive|negative"}}]'),
])
