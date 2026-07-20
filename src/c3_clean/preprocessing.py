"""Immutable preprocessing and Unicode emoji handling for clean C3 runs."""
from __future__ import annotations

import importlib.metadata
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

VARIATION_SELECTORS = ("\ufe0e", "\ufe0f")
SKIN_TONE_CODEPOINTS = range(0x1F3FB, 0x1F400)
ZERO_WIDTH_JOINER = "\u200d"


def emoji_package_version() -> str:
    return importlib.metadata.version("emoji")


def _emoji_module():
    try:
        import emoji
    except ImportError as exc:
        raise RuntimeError(
            "The pinned 'emoji' package is required for preprocessing and coverage audits"
        ) from exc
    return emoji


def extract_emoji_sequence(text: str) -> list[str]:
    """Extract Unicode emoji sequences in order from unmodified raw text.

    ``emoji.emoji_list`` preserves ZWJ sequences and skin-tone modifiers as
    components of a single returned sequence. Repeated occurrences are kept.
    Text aliases such as ``:smile:`` are not converted.
    """
    emoji = _emoji_module()
    return [item["emoji"] for item in emoji.emoji_list(str(text))]


def _remove_duplicate_emoji_runs(text: str) -> str:
    """Collapse only adjacent, identical Unicode emoji sequences in model text."""
    emoji = _emoji_module()
    matches = emoji.emoji_list(text)
    if not matches:
        return text
    output: list[str] = []
    cursor = 0
    previous_token: str | None = None
    previous_end = -1
    for item in matches:
        start, end, token = item["match_start"], item["match_end"], item["emoji"]
        between = text[cursor:start]
        if between:
            previous_token = None
        output.append(between)
        if not (start == previous_end and token == previous_token):
            output.append(token)
        previous_token = token
        previous_end = end
        cursor = end
    output.append(text[cursor:])
    return "".join(output)


def _remove_duplicate_alpha_chars(text: str) -> str:
    previous: str | None = None
    output: list[str] = []
    for character in text:
        if character.isalpha() and character == previous:
            continue
        output.append(character)
        previous = character
    return "".join(output)


def _read_teencode(path: Path) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return tuple(pairs)


@dataclass(frozen=True)
class ImmutablePreprocessor:
    """The single preprocessing function used by every controlled experiment."""

    patterns: tuple[tuple[str, str], ...]
    teencode: tuple[tuple[str, str], ...]
    version: str = "c3_clean_v1"

    @classmethod
    def from_docs(cls, docs_dir: str | Path) -> "ImmutablePreprocessor":
        docs_dir = Path(docs_dir)
        patterns_path = docs_dir / "patterns.json"
        teencode_path = docs_dir / "teencode4.txt"
        if not patterns_path.is_file() or not teencode_path.is_file():
            raise FileNotFoundError(
                f"Expected patterns.json and teencode4.txt under {docs_dir.resolve()}"
            )
        with patterns_path.open("r", encoding="utf-8") as handle:
            patterns: Mapping[str, str] = json.load(handle)
        return cls(tuple(patterns.items()), _read_teencode(teencode_path))

    def __call__(self, original_text: str) -> str:
        text = str(original_text).lower()
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        text = _remove_duplicate_alpha_chars(text)
        text = _remove_duplicate_emoji_runs(text)
        for old, new in self.teencode:
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        text = re.sub(r"(?<![.,!?;:])\n", ". ", text)
        text = re.sub(r"\n([.,!?;:])?", r" \1", text)
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        return re.sub(r"\s+", " ", text).strip()


def prepare_text_columns(
    frame: pd.DataFrame, preprocessor: ImmutablePreprocessor
) -> pd.DataFrame:
    """Return a copy with immutable ``original_text`` and derived ``model_text``."""
    if "text" not in frame:
        raise ValueError("Input frame must contain a text column")
    output = frame.copy(deep=True)
    output["original_text"] = output["text"].astype(str)
    output["model_text"] = output["original_text"].map(preprocessor)
    return output


def token_length_statistics(texts: Iterable[str], tokenizer: Any) -> dict[str, float | int]:
    lengths = np.asarray(
        [
            len(
                tokenizer(
                    str(text),
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )["input_ids"]
            )
            for text in texts
        ],
        dtype=np.int64,
    )
    if lengths.size == 0:
        raise ValueError("Cannot compute token length statistics for an empty split")
    return {
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "maximum": int(lengths.max()),
    }


def emoji_presence_statistics(texts: Sequence[str]) -> dict[str, Any]:
    counts = np.asarray([bool(extract_emoji_sequence(text)) for text in texts], dtype=bool)
    return {
        "comments": int(counts.size),
        "comments_with_unicode_emoji": int(counts.sum()),
        "percentage_with_unicode_emoji": float(100.0 * counts.mean()) if counts.size else 0.0,
        "emoji_package_version": emoji_package_version(),
    }


def _lookup_candidates(token: str) -> tuple[str, ...]:
    """Lookup exact token, then a token without text/emoji variation selectors."""
    without_vs = token
    for selector in VARIATION_SELECTORS:
        without_vs = without_vs.replace(selector, "")
    return (token,) if without_vs == token else (token, without_vs)


def resolve_emoji2vec_key(token: str, keyed_vectors: Any) -> str | None:
    for candidate in _lookup_candidates(token):
        if candidate in keyed_vectors:
            return candidate
    return None


def emoji_vector_for_text(
    original_text: str,
    keyed_vectors: Any,
    *,
    dimension: int = 300,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    covered_count = 0
    for token in extract_emoji_sequence(original_text):
        key = resolve_emoji2vec_key(token, keyed_vectors)
        if key is not None:
            vectors.append(np.asarray(keyed_vectors[key], dtype=np.float32))
            covered_count += 1
        else:
            vectors.append(np.zeros(dimension, dtype=np.float32))
    if covered_count == 0:
        return np.zeros(dimension, dtype=np.float32)
    result = np.stack(vectors).mean(axis=0).astype(np.float32)
    if result.shape != (dimension,):
        raise ValueError(f"Emoji2Vec vector shape {result.shape} != ({dimension},)")
    return result


def emoji2vec_coverage(
    texts: Sequence[str], keyed_vectors: Any, *, split: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return occurrence/unique coverage and OOV frequency without raw text."""
    occurrences: Counter[str] = Counter()
    covered_occurrences = 0
    comments_with_covered = 0
    comments_only_oov = 0
    for text in texts:
        tokens = extract_emoji_sequence(text)
        occurrences.update(tokens)
        covered = [resolve_emoji2vec_key(token, keyed_vectors) is not None for token in tokens]
        comments_with_covered += int(any(covered))
        comments_only_oov += int(bool(tokens) and not any(covered))

    key_by_token = {
        token: resolve_emoji2vec_key(token, keyed_vectors) for token in occurrences
    }
    covered_occurrences = sum(
        count for token, count in occurrences.items() if key_by_token[token] is not None
    )
    unique_covered = sum(key is not None for key in key_by_token.values())
    total_occurrences = sum(occurrences.values())
    unique_count = len(occurrences)
    summary = {
        "split": split,
        "comments": len(texts),
        "total_emoji_occurrences": total_occurrences,
        "unique_emoji_count": unique_count,
        "covered_occurrences": covered_occurrences,
        "oov_occurrences": total_occurrences - covered_occurrences,
        "occurrence_coverage_percentage": (
            100.0 * covered_occurrences / total_occurrences if total_occurrences else 0.0
        ),
        "covered_unique_emoji_count": unique_covered,
        "unique_coverage_percentage": (
            100.0 * unique_covered / unique_count if unique_count else 0.0
        ),
        "comments_with_at_least_one_covered_emoji": comments_with_covered,
        "comments_with_at_least_one_covered_emoji_percentage": (
            100.0 * comments_with_covered / len(texts) if texts else 0.0
        ),
        "comments_with_only_oov_emojis": comments_only_oov,
        "comments_with_only_oov_emojis_percentage": (
            100.0 * comments_only_oov / len(texts) if texts else 0.0
        ),
        "lookup_policy": "exact_then_strip_variation_selectors",
        "skin_tone_policy": "preserve_in_sequence_no_base_fallback",
        "zwj_policy": "preserve_complete_sequence_no_component_fallback",
        "repeated_emoji_policy": "count_each_occurrence",
        "alias_policy": "do_not_convert_text_aliases",
        "oov_vector_policy": "zero_vector_participates_in_occurrence_mean",
    }
    oov_rows = [
        {
            "split": split,
            "emoji": token,
            "occurrences": count,
            "contains_variation_selector": any(vs in token for vs in VARIATION_SELECTORS),
            "contains_skin_tone_modifier": any(ord(ch) in SKIN_TONE_CODEPOINTS for ch in token),
            "contains_zwj": ZERO_WIDTH_JOINER in token,
        }
        for token, count in occurrences.most_common()
        if key_by_token[token] is None
    ]
    return summary, pd.DataFrame(oov_rows)


__all__ = [
    "ImmutablePreprocessor",
    "emoji2vec_coverage",
    "emoji_package_version",
    "emoji_presence_statistics",
    "emoji_vector_for_text",
    "extract_emoji_sequence",
    "prepare_text_columns",
    "resolve_emoji2vec_key",
    "token_length_statistics",
]
