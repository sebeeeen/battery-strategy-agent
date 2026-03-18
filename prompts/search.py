"""
Search Agent 프롬프트 — 확증 편향 방지를 위한 긍정/부정 쌍 쿼리 전략.

- 각 주제에 대해 긍정적 측면 쿼리 + 부정적/비판적 측면 쿼리를 쌍으로 생성
- 두 관점의 검색 결과를 모두 수집해 균형 잡힌 정보 확보
"""

from langchain_core.prompts import ChatPromptTemplate

# Query Pair Generator

SEARCH_QUERY_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research analyst tasked with generating balanced search queries. "
     "For each topic, generate EXACTLY TWO queries:\n"
     "1. POSITIVE query: focuses on strengths, successes, opportunities\n"
     "2. NEGATIVE query: focuses on weaknesses, challenges, risks, criticisms\n\n"
     "This dual-query approach prevents confirmation bias.\n\n"
     "Return JSON format:\n"
     "{{\n"
     '  "positive_query": "...",\n'
     '  "negative_query": "..."\n'
     "}}"),
    ("human",
     "Topic: {topic}\n"
     "Context: Analyzing battery company portfolio diversification strategies in 2024-2025.\n\n"
     "Generate balanced search query pair:"),
])

# Predefined Balanced Query Pairs
# 확증 편향 방지를 위한 사전 정의 긍정/부정 쌍 쿼리

BALANCED_QUERY_PAIRS = [
    # LGES vs CATL ESS + 포트폴리오 (통합)
    {
        "topic": "LGES CATL ESS strategy comparison",
        "positive": "LG Energy Solution CATL ESS energy storage strategy growth 2024 2025 comparison",
        "negative": "LG Energy Solution CATL ESS challenges overcapacity financial pressure 2024",
    },
    # 기술 차별화 (LFP, 나트륨이온, 포트폴리오 다각화)
    {
        "topic": "LGES CATL technology differentiation",
        "positive": "CATL sodium LFP LG Energy Solution new cathode technology competitive advantage 2024 2025",
        "negative": "CATL LG battery technology risks commercialization challenges geopolitical EV slowdown 2024",
    },
    # 배터리 시장 배경
    {
        "topic": "battery market background",
        "positive": "global battery ESS EV market growth opportunity 2024 2025 clean energy transition",
        "negative": "EV chasm slowdown battery overcapacity price war demand uncertainty 2024",
    },
]


# Search Result Summarization

SEARCH_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research analyst. Summarize the web search results for a battery industry report. "
     "Extract only factual, verifiable information. "
     "Note the source URL and date for each key point. "
     "Write in Korean."),
    ("human",
     "Topic: {topic}\n"
     "Query type: {query_type}\n\n"
     "Search Results:\n{results}\n\n"
     "Please summarize the key findings, focusing on facts relevant to battery company strategy analysis:"),
])
