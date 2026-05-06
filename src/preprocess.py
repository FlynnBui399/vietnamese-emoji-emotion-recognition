"""Text preprocessing pipeline matching the ViGoEmotions Phase 1 baseline.

Mirrors `clean_text` from `ViGoEmotions/model/ViSoBERT.ipynb`:

    text -> lower
         -> normalize_pattern(text, pattern_dict)   # e.g. ":))))" -> ":)"
         -> remove_duplicate_chars                 # "cuườiii" -> "cuười"
         -> remove_duplicate_emoji                 # "😄😄😄" -> "😄"
         -> replace_teencode(text, teen_dict)      # whole-word teen -> normal
         -> newline / punctuation spacing
         -> collapse repeated whitespace

`replacing_emojis` is intentionally NOT part of `clean_text` because the
baseline notebook keeps that line commented out (Scenario 1 keeps raw emoji).

Resources are loaded from `docs/` (configurable via `TrainConfig.docs_dir`).
`load_resources` raises FileNotFoundError if the directory or required files
are missing so training cannot silently skip baseline normalization steps.

    docs/patterns.json   {"<regex>": "<replacement>", ...}   (required)
    docs/teencode4.txt   one teencode pair per line, "<teen>\\t<normal>" (required)
    docs/emojis.json     {"<emoji>": "<word>", ...}            (optional; unused by baseline Scenario 1)
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import emoji as _emoji_lib

from .utils import get_logger

LOGGER = get_logger(__name__)


def get_pyvi_segmenter() -> Callable[[str], str] | None:
    """Return `pyvi.ViTokenizer.tokenize` if available, else None.

    The baseline notebook applies Vietnamese word segmentation (e.g.
    "Tôi rất vui" -> "Tôi rất_vui") *after* clean_text and *before* the
    SentencePiece tokenizer. Returning None lets the caller skip this step
    cleanly when `pyvi` isn't installed.
    """
    try:
        # pyvi unpickles a CRF model; NumPy 2.4+ warns on legacy dtype(align=0) in the blob.
        try:
            from numpy.exceptions import VisibleDeprecationWarning as _NumpyVisDep
        except ImportError:
            import numpy as _np

            _NumpyVisDep = getattr(_np, "VisibleDeprecationWarning", DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _NumpyVisDep)
            from pyvi.ViTokenizer import tokenize
    except ImportError:
        LOGGER.warning(
            "pyvi is not installed; Vietnamese word segmentation will be skipped. "
            "Install it with `pip install pyvi` to fully match the baseline."
        )
        return None
    return tokenize


@dataclass
class Resources:
    pattern_dict: dict[str, str] = field(default_factory=dict)
    emoji_dict: dict[str, str] = field(default_factory=dict)
    teen_dict: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.pattern_dict or self.emoji_dict or self.teen_dict)


def _read_teencode(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def load_resources(docs_dir: str | Path | None) -> Resources:
    """Load `pattern_dict`, `emoji_dict`, `teen_dict` for `clean_text`.

    Raises:
        FileNotFoundError: If ``docs_dir`` is missing/empty, is not a directory,
            or required files ``patterns.json`` / ``teencode4.txt`` are absent.
    """
    if not docs_dir:
        raise FileNotFoundError(
            "docs_dir is required but was empty or None; expected a directory "
            "containing patterns.json and teencode4.txt."
        )

    docs = Path(docs_dir)
    if not docs.is_dir():
        raise FileNotFoundError(
            f"docs_dir does not exist or is not a directory: {docs.resolve()}"
        )

    patterns_path = docs / "patterns.json"
    if not patterns_path.is_file():
        raise FileNotFoundError(
            f"Missing required preprocessing file: {patterns_path} "
            "(baseline pattern normalization)."
        )

    teen_path = docs / "teencode4.txt"
    if not teen_path.is_file():
        raise FileNotFoundError(
            f"Missing required preprocessing file: {teen_path} "
            "(baseline teencode replacement)."
        )

    res = Resources()
    with patterns_path.open("r", encoding="utf-8") as f:
        res.pattern_dict = json.load(f) or {}

    emojis_path = docs / "emojis.json"
    if emojis_path.is_file():
        with emojis_path.open("r", encoding="utf-8") as f:
            res.emoji_dict = json.load(f) or {}

    res.teen_dict = _read_teencode(teen_path)

    LOGGER.info(
        "Loaded preprocessing resources: %d patterns, %d emoji entries, %d teencode pairs",
        len(res.pattern_dict),
        len(res.emoji_dict),
        len(res.teen_dict),
    )
    return res


def normalize_pattern(text: str, pattern_dict: Mapping[str, str]) -> str:
    for pattern, replacement in pattern_dict.items():
        text = re.sub(pattern=pattern, repl=replacement, string=text)
    return text


def remove_duplicate_chars(text: str) -> str:
    """Collapse runs of repeated alphabetic chars (e.g. 'ơiiiii' -> 'ơi')."""
    prev = None
    out: list[str] = []
    for ch in text:
        if ch.isalpha() and prev == ch:
            continue
        prev = ch
        out.append(ch)
    return "".join(out)


def remove_duplicate_emoji(text: str) -> str:
    """Collapse runs of identical emoji glyphs (e.g. '😄😄😄' -> '😄')."""
    out: list[str] = []
    prev = None
    for ch in text:
        if ch in _emoji_lib.EMOJI_DATA:
            if ch == prev:
                continue
            prev = ch
        else:
            prev = None
        out.append(ch)
    return "".join(out)


def replace_teencode(text: str, teen_dict: Mapping[str, str]) -> str:
    for old, new in teen_dict.items():
        pattern = re.compile(r"\b{}\b".format(re.escape(old)))
        text = pattern.sub(new, text)
    return text


def replacing_emojis(text: str, emoji_dict: Mapping[str, str]) -> str:
    """Replace emoji glyphs with words. NOT applied by baseline `clean_text`."""
    for em, word in emoji_dict.items():
        text = text.replace(em, " " + word + " ")
    return text


_PUNCT_RE = re.compile(r"([.,!?;:])")
_NEWLINE_NOPUNCT_RE = re.compile(r"(?<![.,!?;:])\n")
_NEWLINE_PUNCT_RE = re.compile(r"\n([.,!?;:])?")
_WS_RE = re.compile(r"\s+")


def build_preprocessor(
    *,
    apply_clean_text: bool,
    apply_pyvi: bool,
    docs_dir: str | Path | None,
) -> Callable[[str], str] | None:
    """Return a single ``str -> str`` that runs ``clean_text`` then optional pyvi.

    Composes all enabled steps in order so callers pass **one** callable to
    ``load_split`` / ``build_dataloaders`` — never a list of functions.
    """
    steps: list[Callable[[str], str]] = []
    if apply_clean_text:
        resources = load_resources(docs_dir)
        steps.append(make_clean_text(resources))
        LOGGER.info("clean_text enabled (docs_dir=%s)", docs_dir)
    else:
        LOGGER.info("clean_text disabled (apply_clean_text=False)")

    if apply_pyvi:
        pyvi_seg = get_pyvi_segmenter()
        if pyvi_seg is not None:
            steps.append(pyvi_seg)
            LOGGER.info("pyvi word segmentation enabled (after clean_text)")
    else:
        LOGGER.info("pyvi disabled (apply_pyvi=False)")

    if not steps:
        return None

    def preprocess(text: str) -> str:
        for fn in steps:
            text = fn(text)
        return text

    return preprocess


def make_clean_text(resources: Resources) -> Callable[[str], str]:
    """Return a `clean_text(str) -> str` closure bound to the given resources."""
    pattern_dict = resources.pattern_dict
    teen_dict = resources.teen_dict

    def clean_text(text: str) -> str:
        text = text.lower()
        text = normalize_pattern(text, pattern_dict)
        text = remove_duplicate_chars(text)
        text = remove_duplicate_emoji(text)
        text = replace_teencode(text, teen_dict)

        # Replace newline with ". " when no punctuation precedes it.
        text = _NEWLINE_NOPUNCT_RE.sub(". ", text)
        # Strip remaining \n adjacent to punctuation.
        text = _NEWLINE_PUNCT_RE.sub(r" \1", text)
        # Insert spaces around ASCII punctuation.
        text = _PUNCT_RE.sub(r" \1 ", text)
        # Collapse repeated whitespace.
        text = _WS_RE.sub(" ", text).strip()
        return text

    return clean_text


__all__ = [
    "Resources",
    "build_preprocessor",
    "get_pyvi_segmenter",
    "load_resources",
    "make_clean_text",
    "normalize_pattern",
    "remove_duplicate_chars",
    "remove_duplicate_emoji",
    "replace_teencode",
    "replacing_emojis",
]
