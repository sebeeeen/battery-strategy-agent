"""
RAG Agent A(LGES) / RAG Agent B(CATL) 프롬프트 템플릿.

교재 Self-Reflection 패턴 적용:
  Draft Generation → Self-Reflection (APPROVED / REVISE / RETRIEVE)
"""

from langchain_core.prompts import ChatPromptTemplate

# Context Extraction Prompt

RAG_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strategic analyst specializing in the battery industry. "
     "Based on the provided document context, extract key information about "
     "the company's portfolio diversification strategy. "
     "Focus on: business directions, competitive advantages, ESS strategy, "
     "EV battery strategy, new businesses, financial performance, and market positioning.\n\n"
     "Extract structured information in Korean. Be specific and cite data/numbers where available."),
    ("human",
     "Company: {company}\n"
     "Analysis Topic: {topic}\n\n"
     "Document Context:\n{context}\n\n"
     "Please extract and summarize the key strategic information relevant to the topic above. "
     "Structure your response with clear sections."),
])


# Self-Reflection Prompt

RAG_SELF_REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a quality evaluator for a battery industry strategy report. "
     "Evaluate the following draft analysis on three criteria:\n"
     "1. FACTUAL_GROUNDING: Is every claim supported by the provided context?\n"
     "2. COMPLETENESS: Does it cover the key aspects of the topic?\n"
     "3. SPECIFICITY: Does it include specific data, numbers, or examples?\n\n"
     "Return a JSON verdict:\n"
     "{{\n"
     '  "verdict": "APPROVED" | "REVISE" | "RETRIEVE",\n'
     '  "reason": "brief explanation",\n'
     '  "revision_guidance": "what to improve (if REVISE)",\n'
     '  "missing_info": "what info is needed (if RETRIEVE)"\n'
     "}}\n\n"
     "Verdict meanings:\n"
     "- APPROVED: draft is complete, factual, and specific\n"
     "- REVISE: draft needs improvement but context is sufficient\n"
     "- RETRIEVE: additional retrieval is needed due to missing information"),
    ("human",
     "Topic: {topic}\n"
     "Company: {company}\n\n"
     "Draft Analysis:\n{draft}\n\n"
     "Provided Context:\n{context}\n\n"
     "Evaluate this draft:"),
])


# Draft Revision Prompt

RAG_REVISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a battery industry strategy analyst. "
     "Improve the draft analysis based on the revision guidance provided. "
     "Use only the information in the provided context. "
     "Write in Korean."),
    ("human",
     "Company: {company}\n"
     "Topic: {topic}\n\n"
     "Current Draft:\n{draft}\n\n"
     "Revision Guidance: {guidance}\n\n"
     "Document Context:\n{context}\n\n"
     "Please provide an improved analysis:"),
])
