"""
src/generator.py
Agentic LLM generation layer using Groq for ultra-fast inference.

Architecture:
  Single-Pass Chain-of-Thought (CoT) Agent → Extracts, verifies, and formats
  BIS standards in one strict LLM pass to eliminate double-network latency.
"""

import json
import logging
import re
from typing import Optional

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

COT_PROMPT = """You are a strict compliance parser. First, extract potential BIS standards from the context. Second, verify internally that the standard is explicitly mentioned in the text. Finally, output ONLY the verified standards in valid JSON format. If none are verified, output an empty array.

Your output MUST be a valid JSON object matching this schema exactly:
{
  "verified": [
    {"id": "IS 269: 1989", "rationale": "Explain why it applies", "confidence": 0.95}
  ],
  "dropped": [
    {"id": "IS 999: 2000", "reason": "Not found in context"}
  ]
}
Do not output any markdown formatting or extra text. ONLY output the JSON object.
"""

STANDARD_EXTRACT_PATTERN = re.compile(
    r"IS\s+\d{1,5}(?:\s*(?:\(.*?\))?\s*:\s*\d{4})?", re.IGNORECASE
)


def _call_groq(system_prompt: str, user_message: str, model: str) -> str:
    """Make a single Groq API call. Returns raw text."""
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content or "{}"


def _build_context_block(context_chunks: list[dict]) -> str:
    """Build a compact context string from retrieved chunks."""
    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        std_id = chunk.get("metadata", {}).get("standard_id", "Unknown")
        text = chunk.get("text", "")[:1500]
        parts.append(f"--- Chunk {i} (ID: {std_id}) ---\n{text}")
    return "\n\n".join(parts)


def _parse_json_safe(raw: str, fallback=None):
    """Parse JSON from LLM output with multiple fallback strategies."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    json_match = re.search(r"[\[\{].*[\]\}]", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return fallback


def _run_single_pass_agent(query: str, context_block: str, model: str) -> dict:
    """Run the single-pass CoT agent to extract and verify."""
    user_msg = (
        f"Product/Material Query: {query}\n\n"
        f"Context Chunks:\n{context_block}\n\n"
        f"Extract and verify the standards."
    )
    raw = _call_groq(COT_PROMPT, user_msg, model)
    logger.info("[AGENT CoT] Raw output: %s", raw[:200])
    parsed = _parse_json_safe(raw, fallback={"verified": [], "dropped": []})
    if isinstance(parsed, dict) and "verified" in parsed:
        return parsed
    return {"verified": [], "dropped": []}


def generate_response(
    query: str,
    context_chunks: list[dict],
    model: Optional[str] = None,
) -> list[str]:
    """
    Single-pass Agent generation pipeline.
    Returns a list of standard ID strings, e.g. ["IS 269: 1989"].
    Returns an empty list on any failure (zero-crash policy).
    """
    model = model or GROQ_MODEL
    context_block = _build_context_block(context_chunks)
    try:
        result = _run_single_pass_agent(query, context_block, model)
        verified = result.get("verified", [])

        standards = []
        for item in verified:
            if isinstance(item, dict):
                std_id = item.get("id", "")
                if std_id:
                    standards.append(std_id)
            elif isinstance(item, str):
                standards.append(item)

        seen = set()
        unique = []
        for s in standards:
            normalized = re.sub(r"\s+", " ", s.strip())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        logger.info("[AGENT] Returned %d verified standards.", len(unique))
        return unique[:5]
    except Exception as e:
        logger.error("Agent pipeline failed: %s", e, exc_info=True)
        fallback = []
        for chunk in context_chunks:
            std_id = chunk.get("metadata", {}).get("standard_id")
            if std_id and std_id not in fallback and not std_id.startswith("UNKNOWN"):
                fallback.append(std_id)
        logger.info("Falling back to metadata extraction: %d standards", len(fallback))
        return fallback[:5]


def generate_response_detailed(
    query: str,
    context_chunks: list[dict],
    model: Optional[str] = None,
) -> dict:
    """
    Same as generate_response but returns the full verification report.
    Used by the Streamlit UI for richer display.
    """
    model = model or GROQ_MODEL
    context_block = _build_context_block(context_chunks)
    try:
        result = _run_single_pass_agent(query, context_block, model)
        verified = result.get("verified", [])
        dropped = result.get("dropped", [])

        clean_verified = []
        for item in verified:
            if isinstance(item, dict) and item.get("id"):
                clean_verified.append(
                    {
                        "id": re.sub(r"\s+", " ", item["id"].strip()),
                        "rationale": item.get("rationale", "N/A"),
                        "confidence": item.get("confidence", 0.85),
                    }
                )
        return {
            "verified": clean_verified[:5],
            "dropped": dropped,
            "extractor_count": len(clean_verified) + len(dropped),
            "verifier_count": len(clean_verified),
        }
    except Exception as e:
        logger.error("Detailed generation failed: %s", e, exc_info=True)
        fallback = []
        for chunk in context_chunks:
            std_id = chunk.get("metadata", {}).get("standard_id")
            if std_id and not std_id.startswith("UNKNOWN"):
                fallback.append(
                    {
                        "id": std_id,
                        "rationale": "Extracted from retrieval metadata",
                        "confidence": 0.5,
                    }
                )
        return {
            "verified": fallback[:5],
            "dropped": [],
            "extractor_count": 0,
            "verifier_count": len(fallback),
        }
