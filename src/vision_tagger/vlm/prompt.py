"""Fixed prompt for the VLM stage. Any change here changes the stage version."""

from __future__ import annotations

import hashlib

from .schema import Vocabulary

PROMPT_REVISION = "1"

SYSTEM_TEMPLATE = """You are a meticulous image cataloguing assistant for a private photo archive.
Analyse the image and answer ONLY with a JSON object that follows the given schema.

Rules:
- Describe only what is actually visible. Do not guess names of people.
- For fields with allowed values, use exactly one of the allowed values (lowercase).
  Use "unknown", "other" or "none" when nothing fits or you are not sure.
- Lists contain singular, lowercase, common English nouns (e.g. "car", "coffee cup"),
  most important first, no duplicates. Use an empty list if nothing applies.
- Fields marked as "assume" are your best judgement about the image's purpose; give an honest
  confidence between 0 and 1 in the matching *_confidence field.
- Be consistent: similar images must get the same values.

Fields:
{field_guide}
"""

USER_PROMPT = "Catalogue this image. Answer with the JSON object only."


def system_prompt(vocab: Vocabulary) -> str:
    return SYSTEM_TEMPLATE.format(field_guide=vocab.field_guide())


def prompt_version(vocab: Vocabulary) -> str:
    blob = (PROMPT_REVISION + system_prompt(vocab) + USER_PROMPT + vocab.fingerprint).encode()
    return f"p{PROMPT_REVISION}-{hashlib.sha1(blob).hexdigest()[:8]}"
