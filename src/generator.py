"""
src/generator.py
────────────────
Dual-Agent LLM generation layer using Groq for ultra-fast inference.

Architecture:
  Agent 1 (Extractor) → Extracts Standard IDs + Rationales from context
  Agent 2 (Verifier)  → Cross-references against original chunks,
                         strips hallucinations, validates rationales

Both agents use Groq (Llama-3) to maintain < 5s total latency.
"""

import json
import logging
import re
from typing import Optional

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# ── Agent 1: Extractor Prompt ────────────────────────────────────────────────
EXTRACTOR_PROMPT = """You are a BIS (Bureau of Indian Standards) compliance extractor.
Analyze the provided context chunks about Indian Standards for building materials.
Extract ALL relevant BIS standard IDs that apply to the user's query.

For EACH standard, provide:
1. The exact standard ID as it appears (e.g., "IS 269: 1989")
2. A one-line rationale explaining WHY it applies to the query

RULES:
- Extract ONLY standards that appear in the provided context.
- Do NOT invent or fabricate any standard IDs.
- Be thorough — extract every relevant standard from the context.

Respond with ONLY a valid JSON array of objects. No markdown, no explanation.
Example: [{"id": "IS 269: 1989", "rationale": "Covers OPC specifications for construction"}]
"""

# ── Agent 2: Verifier Prompt ─────────────────────────────────────────────────
VERIFIER_PROMPT = """You are a strict BIS compliance verifier. Your job is to VERIFY
the extracted standards against the original source context.

You will receive:
1. The original query
2. The extracted standards (from Agent 1)
3. The original context chunks

VERIFICATION RULES:
- CHECK that each standard ID actually exists verbatim in the context chunks.
- CHECK that the rationale is factually supported by the context.
- REMOVE any standard that does NOT appear in the context (hallucination).
- REMOVE any standard with a weak or unsupported rationale.
- Keep only genuinely relevant, verified standards.

Respond with ONLY a valid JSON object:
{
  "verified": [{"id": "IS 269: 1989", "rationale": "...", "confidence": 0.95}],
  "dropped": [{"id": "IS 999: 2000", "reason": "Not found in context"}]
}
"""

# ── Regex fallback ───────────────────────────────────────────────────────────
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
    )
    return completion.choices[0].message.content or ""


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
    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting JSON from markdown
    json_match = re.search(r"[\[\{].*[\]\}]", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return fallback


def _extract_standards(
    query: str, context_block: str, model: str
) -> list[dict]:
    """Agent 1: Extract standard IDs and rationales from context."""
    user_msg = (
        f"Product/Material Query: {query}\n\n"
        f"Context Chunks:\n{context_block}\n\n"
        f"Extract all applicable BIS standards from the above context."
    )

    raw = _call_groq(EXTRACTOR_PROMPT, user_msg, model)
    logger.info("[AGENT-1 Extractor] Raw output: %s", raw[:200])

    parsed = _parse_json_safe(raw, fallback=[])
    if isinstance(parsed, list):
        return parsed
    return []


def _verify_standards(
    query: str,
    extracted: list[dict],
    context_block: str,
    model: str,
) -> dict:
    """Agent 2: Verify extracted standards against original context."""
    user_msg = (
        f"Original Query: {query}\n\n"
        f"Extracted Standards (from Agent 1):\n{json.dumps(extracted, indent=2)}\n\n"
        f"Original Context Chunks:\n{context_block}\n\n"
        f"Verify each standard. Remove hallucinations. Return JSON."
    )

    raw = _call_groq(VERIFIER_PROMPT, user_msg, model)
    logger.info("[AGENT-2 Verifier] Raw output: %s", raw[:200])

    parsed = _parse_json_safe(raw, fallback=None)
    if isinstance(parsed, dict) and "verified" in parsed:
        return parsed
    # Fallback: treat the whole response as the verified list
    if isinstance(parsed, list):
        return {"verified": parsed, "dropped": []}
    return {"verified": extracted, "dropped": []}


def generate_response(
    query: str,
    context_chunks: list[dict],
    model: Optional[str] = None,
) -> list[str]:
    """
    Dual-Agent generation pipeline:
      Agent 1 (Extractor) → Agent 2 (Verifier) → final standards list.

    Returns a list of standard ID strings, e.g. ["IS 269: 1989"].
    Returns an empty list on any failure (zero-crash policy).
    """
    model = model or GROQ_MODEL
    context_block = _build_context_block(context_chunks)

    try:
        # ── Agent 1: Extract ─────────────────────────────────────────────
        extracted = _extract_standards(query, context_block, model)
        logger.info(
            "[AGENT-1] Extracted %d candidate standards", len(extracted)
        )

        if not extracted:
            # Regex fallback from context
            found = STANDARD_EXTRACT_PATTERN.findall(context_block)
            if found:
                extracted = [{"id": s.strip(), "rationale": "Regex extraction"} for s in found]
                logger.info("[AGENT-1] Regex fallback: %d standards", len(extracted))

        # ── Agent 2: Verify ──────────────────────────────────────────────
        result = _verify_standards(query, extracted, context_block, model)
        verified = result.get("verified", [])
        dropped = result.get("dropped", [])

        logger.info(
            "[AGENT-2] Verified %d standards. Dropped %d hallucinations.",
            len(verified),
            len(dropped),
        )

        # Extract just the ID strings
        standards = []
        for item in verified:
            if isinstance(item, dict):
                std_id = item.get("id", "")
                if std_id:
                    standards.append(std_id)
            elif isinstance(item, str):
                standards.append(item)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in standards:
            normalized = re.sub(r"\s+", " ", s.strip())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        return unique[:5]

    except Exception as e:
        logger.error("Dual-agent pipeline failed: %s", e, exc_info=True)
        # Zero-crash fallback: extract from metadata
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
    Same as generate_response but returns the full verification report
    including rationales, confidence scores, and dropped hallucinations.
    Used by the Streamlit UI for richer display.
    """
    model = model or GROQ_MODEL
    context_block = _build_context_block(context_chunks)

    try:
        extracted = _extract_standards(query, context_block, model)

        if not extracted:
            found = STANDARD_EXTRACT_PATTERN.findall(context_block)
            if found:
                extracted = [{"id": s.strip(), "rationale": "Regex extraction"} for s in found]

        result = _verify_standards(query, extracted, context_block, model)
        verified = result.get("verified", [])
        dropped = result.get("dropped", [])

        # Normalize verified entries
        clean_verified = []
        for item in verified:
            if isinstance(item, dict) and item.get("id"):
                clean_verified.append({
                    "id": re.sub(r"\s+", " ", item["id"].strip()),
                    "rationale": item.get("rationale", "N/A"),
                    "confidence": item.get("confidence", 0.85),
                })

        return {
            "verified": clean_verified[:5],
            "dropped": dropped,
            "extractor_count": len(extracted),
            "verifier_count": len(clean_verified),
        }

    except Exception as e:
        logger.error("Detailed generation failed: %s", e, exc_info=True)
        fallback = []
        for chunk in context_chunks:
            std_id = chunk.get("metadata", {}).get("standard_id")
            if std_id and not std_id.startswith("UNKNOWN"):
                fallback.append({
                    "id": std_id,
                    "rationale": "Extracted from retrieval metadata",
                    "confidence": 0.5,
                })
        return {
            "verified": fallback[:5],
            "dropped": [],
            "extractor_count": 0,
            "verifier_count": len(fallback),
        }
