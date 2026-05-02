"""
src/generator.py
────────────────
LLM generation layer using Groq for ultra-fast inference.
Takes retrieved context chunks and produces a structured list
of applicable BIS standard IDs.
"""

import json
import logging
import re
from typing import Optional

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# ── System prompt (strict compliance parser) ─────────────────────────────────
SYSTEM_PROMPT = """You are a strict BIS (Bureau of Indian Standards) compliance parser.
Your task is to analyze the provided context about Indian Standards for building materials 
and extract ONLY the relevant BIS standard IDs that are applicable to the user's query.

RULES:
1. Extract standard IDs EXACTLY as they appear in the context (e.g., "IS 269: 1989").
2. Do NOT invent, fabricate, or hallucinate any standard IDs that are not present in the context.
3. Return ONLY standards that are genuinely relevant to the query.
4. If no relevant standards are found, return an empty list.
5. Respond with ONLY a valid JSON array of standard ID strings. No explanation, no markdown.

Example output: ["IS 269: 1989", "IS 455: 1989", "IS 383: 1970"]
"""

# ── Regex to extract standard IDs from LLM output as fallback ────────────────
STANDARD_EXTRACT_PATTERN = re.compile(
    r"IS\s+\d{1,5}(?:\s*:\s*\d{4})?", re.IGNORECASE
)


def _parse_llm_output(raw: str) -> list[str]:
    """
    Parse the LLM response into a list of standard ID strings.
    Tries JSON parsing first, then falls back to regex extraction.
    """
    raw = raw.strip()

    # Attempt 1: direct JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: extract JSON array from within markdown/text
    json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [str(s) for s in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 3: regex fallback — grab any "IS XXXX: YYYY" patterns
    found = STANDARD_EXTRACT_PATTERN.findall(raw)
    if found:
        # Normalize whitespace
        return [re.sub(r"\s+", " ", s.strip()) for s in found]

    return []


def generate_response(
    query: str,
    context_chunks: list[dict],
    model: Optional[str] = None,
) -> list[str]:
    """
    Generate a list of applicable BIS standard IDs using the Groq LLM.

    Args:
        query: The user's product description / search query.
        context_chunks: Retrieved chunks from the hybrid search pipeline.
                        Each dict must contain at minimum a "text" key.
        model: Override the default Groq model.

    Returns:
        A list of standard ID strings, e.g. ["IS 269: 1989", "IS 455: 1989"].
        Returns an empty list on any failure (zero-crash policy).
    """
    model = model or GROQ_MODEL

    # Build the context block from retrieved chunks
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        std_id = chunk.get("metadata", {}).get("standard_id", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(
            f"--- Standard Chunk {i} (ID: {std_id}) ---\n{text[:2000]}"
        )
    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Product/Material Query: {query}\n\n"
        f"Retrieved Context:\n{context_block}\n\n"
        f"Based on the above context, return a JSON array of the applicable "
        f"BIS standard IDs for the given query."
    )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw_output = chat_completion.choices[0].message.content or ""
        logger.info("Groq raw output: %s", raw_output[:200])

        standards = _parse_llm_output(raw_output)
        logger.info("Extracted %d standard IDs", len(standards))
        return standards

    except Exception as e:
        logger.error("Groq generation failed: %s", e, exc_info=True)
        # Zero-crash policy: return whatever we can from context metadata
        fallback = []
        for chunk in context_chunks:
            std_id = chunk.get("metadata", {}).get("standard_id")
            if std_id and std_id not in fallback and not std_id.startswith("UNKNOWN"):
                fallback.append(std_id)
        logger.info("Falling back to metadata extraction: %d standards", len(fallback))
        return fallback[:5]
