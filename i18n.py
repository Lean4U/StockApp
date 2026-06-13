"""Output-language translations for the SyntoniaPro dashboard.

The user picks an *insights* language in the sidebar — what they want the
dashboard's labels, narrative copy, and Socratic prompts rendered in.
Whisper still auto-detects the spoken language; this module only governs
how the dashboard talks back to the user.

Translations are intentionally partial right now. English is complete;
Spanish covers the highest-impact UI labels (BASELINE / NOW / OVERLAY /
category chips / section headers) so a Spanish-speaking subject sees
those in their own language. Other languages fall back to English with
a `(English fallback)` marker on the picker. As translations land they
just become new entries in `_T`.
"""
from __future__ import annotations

from typing import Dict, Optional


# Output-language choices presented in the sidebar. Pairs label → Whisper
# language code (for symmetry, even though we currently keep transcription
# on auto-detect for input). None on the value means English.
OUTPUT_LANGUAGES: Dict[str, str] = {
    "English":  "en",
    "Español":  "es",
    "Português": "pt",
    "हिन्दी (Hindi)":   "hi",
    "தமிழ் (Tamil)":   "ta",
    "日本語 (Japanese)": "ja",
    "中文 (Mandarin)":  "zh",
    "العربية (Arabic)":  "ar",
    "Français": "fr",
    "Deutsch":  "de",
    "Italiano": "it",
    "Yorùbá":   "yo",
}


# Translation table. _T[lang_code][key] = translated string.
# Missing keys / languages fall back to English.
_T: Dict[str, Dict[str, str]] = {
    "en": {
        "label.baseline":      "BASELINE",
        "label.now":           "NOW",
        "label.overlay":       "OVERLAY",
        "label.baseline_vs":   "● baseline",
        "label.now_vs":        "● now",
        "label.vs":            "vs",
        "label.stable_zone":   "Stable Zone",
        "label.breach_line":   "BREACH LINE (100% = σ-high)",
        "cat.behavioural":     "Real face change",
        "cat.pose":            "Head movement",
        "cat.hardware":        "Setup drift",
        "cat.other":           "Other",
        "section.where_moved": "Where Your Face Moved",
        "section.over_time":   "How It Changed Over Time",
        "section.nuance":      "NUANCE",
        "section.of":          "OF",
        "field.what_measures": "What it measures.",
        "field.cited_science": "Cited science.",
        "field.plain_lang":    "In plain language.",
        "field.what_said":     "What you said.",
        "severity.subtle":     "SUBTLE",
        "severity.noticeable": "NOTICEABLE",
        "severity.marked":     "MARKED",
        "severity.intense":    "INTENSE",
        "btn.record":          "Record ▶",
        "btn.stop":            "Stop ■",
        "btn.detect":          "Detect events vs selected baseline",
    },
    "es": {
        "label.baseline":      "LÍNEA BASE",
        "label.now":           "AHORA",
        "label.overlay":       "SUPERPOSICIÓN",
        "label.baseline_vs":   "● línea base",
        "label.now_vs":        "● ahora",
        "label.vs":            "vs",
        "label.stable_zone":   "Zona Estable",
        "label.breach_line":   "LÍMITE DE QUIEBRE (100% = σ-alto)",
        "cat.behavioural":     "Cambio facial real",
        "cat.pose":            "Movimiento de cabeza",
        "cat.hardware":        "Deriva de configuración",
        "cat.other":           "Otro",
        "section.where_moved": "Dónde Se Movió Tu Cara",
        "section.over_time":   "Cómo Cambió Con El Tiempo",
        "section.nuance":      "MATIZ",
        "section.of":          "DE",
        "field.what_measures": "Qué mide.",
        "field.cited_science": "Ciencia citada.",
        "field.plain_lang":    "En lenguaje sencillo.",
        "field.what_said":     "Lo que dijiste.",
        "severity.subtle":     "SUTIL",
        "severity.noticeable": "NOTABLE",
        "severity.marked":     "MARCADO",
        "severity.intense":    "INTENSO",
        "btn.record":          "Grabar ▶",
        "btn.stop":            "Parar ■",
        "btn.detect":          "Detectar eventos vs línea base seleccionada",
    },
}


def t(key: str, lang: Optional[str]) -> str:
    """Return the translation of ``key`` in ``lang``. Falls back to English
    when the language or key isn't translated yet.
    """
    if lang and lang in _T and key in _T[lang]:
        return _T[lang][key]
    return _T["en"].get(key, key)


def has_full_translation(lang: Optional[str]) -> bool:
    """Whether the language has a complete-enough translation that the
    sidebar should NOT show a fallback warning."""
    if not lang or lang == "en":
        return True
    if lang not in _T:
        return False
    # Treat a language as complete if its dict has the same keys as English.
    return set(_T[lang].keys()) >= set(_T["en"].keys())
