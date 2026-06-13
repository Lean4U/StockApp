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
        "label.breach_line":   "OUT OF NORM (100%)",
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
        "implication.html":    (
            "Your answer separates <b>context</b> (head pose, lighting, "
            "posture drift — absorbable by the adaptive baseline) from "
            "<b>behaviour</b> (a deliberate expression worth noticing)."
        ),
        "status.out_of_norm":  "🔴 **Out of norm** — your face is significantly off baseline.",
        "status.excursion":    "🟠 **Excursion** — something is shifting.",
        "status.stable":       "🟢 **Stable** — you're within your baseline envelope.",
        "transcript.silence":  "_(silence around this moment)_",
        "msg.stable_take":     (
            "Stable take. No feature crossed the watch line. Your face "
            "stayed within the geometric envelope of your baseline."
        ),
        "spider.subtitle":     (
            "each spoke is one geometric dimension; the green ring is your "
            "stable zone, the red ring is the out-of-norm threshold"
        ),
        "ts.first":            "first at",
        "ts.latest":           "latest at",
        "ts.windows":          "episode(s)",
        "section.baseline_vs": "Baseline vs. now",
        "chart.time":          "time (s)",
        "chart.y_pct":         "% of out-of-norm",
        "chart.legend.baseline":  "BASELINE (0%)",
        "chart.legend.now":       "NOW",
        "chart.legend.watch":     "watch line (±50%)",
        "chart.legend.oon":       "out of norm (±100%)",
        "chart.legend.episode":   "this episode",
        "chart.spider.baseline":  "BASELINE (Stable Zone - Normal)",
        "chart.spider.now":       "NOW (your face at this nuance)",
        # Spider radar axis labels — short plain-English tags.
        "spider.left_brow_height_norm":  "LEFT BROW",
        "spider.right_brow_height_norm": "RIGHT BROW",
        "spider.angle_LM":               "LEFT SMILE",
        "spider.angle_RM":               "RIGHT SMILE",
        "spider.mouth_offset_norm":      "LIP SHIFT",
        "spider.side_left_norm":         "LEFT CHEEK",
        "spider.side_right_norm":        "RIGHT CHEEK",
        "spider.yaw_proxy":              "HEAD TURN",
        "spider.pitch_proxy":            "HEAD NOD",
        "spider.roll_proxy":             "HEAD TILT",
        "spider.eye_mouth_ratio":        "FACE STRETCH",
        "spider.diag_ratio":             "FACE TWIST",
        # Voice-command banner overlays.
        "voice.heard_start":             "HEARD '{w}' — STARTING TAKE…",
        "voice.heard_stop":              "HEARD '{w}' — STOPPING TAKE…",
        # Side / direction labels rendered under the icon column on each
        # inflection row. These were hard-coded English in FEATURE_GLYPH.
        "side_label.left_brow_height_norm":  "◀ LEFT brow",
        "side_label.right_brow_height_norm": "RIGHT brow ▶",
        "side_label.angle_LM":               "◀ LEFT mouth corner",
        "side_label.angle_RM":               "RIGHT mouth corner ▶",
        "side_label.mouth_offset_norm":      "mouth off-centre",
        "side_label.mouth_line_norm":        "mouth width",
        "side_label.side_mouth_norm":        "mouth width",
        "side_label.side_left_norm":         "◀ LEFT side",
        "side_label.side_right_norm":        "RIGHT side ▶",
        "side_label.side_eye_norm":          "eye line / glasses",
        "side_label.eye_line_norm":          "eye line / glasses",
        "side_label.angle_LE":               "◀ LEFT eye corner",
        "side_label.angle_RE":               "RIGHT eye corner ▶",
        "side_label.yaw_proxy":              "◀ head turn ▶",
        "side_label.pitch_proxy":            "▲ chin ▼",
        "side_label.roll_proxy":             "head tilt",
        "side_label.diag_LE_RM_norm":        "LE → RM diagonal",
        "side_label.diag_RE_LM_norm":        "RE → LM diagonal",
        "side_label.diag_ratio":             "diagonal balance",
        "side_label.eye_mouth_ratio":        "vertical proportion",
        "side_label.parallelism_residual":   "non-parallel",
    },
    "es": {
        "label.baseline":      "LÍNEA BASE",
        "label.now":           "AHORA",
        "label.overlay":       "SUPERPOSICIÓN",
        "label.baseline_vs":   "● línea base",
        "label.now_vs":        "● ahora",
        "label.vs":            "vs",
        "label.stable_zone":   "Zona Estable",
        "label.breach_line":   "FUERA DE NORMA (100%)",
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
        "implication.html":    (
            "Tu respuesta separa el <b>contexto</b> (postura de cabeza, "
            "iluminación, deriva postural — absorbible por la línea base "
            "adaptativa) del <b>comportamiento</b> (una expresión "
            "deliberada que vale la pena notar)."
        ),
        "status.out_of_norm":  "🔴 **Fuera de norma** — tu cara está significativamente fuera de la línea base.",
        "status.excursion":    "🟠 **Excursión** — algo está cambiando.",
        "status.stable":       "🟢 **Estable** — estás dentro del rango de tu línea base.",
        "transcript.silence":  "_(silencio alrededor de este momento)_",
        "msg.stable_take":     (
            "Toma estable. Ninguna característica cruzó la línea de vigilancia. "
            "Tu cara se mantuvo dentro del rango geométrico de tu línea base."
        ),
        "spider.subtitle":     (
            "cada radio es una dimensión geométrica; el anillo verde es tu "
            "zona estable, el anillo rojo es el umbral de fuera de norma"
        ),
        "ts.first":            "primera vez a",
        "ts.latest":           "última vez a",
        "ts.windows":          "episodio(s)",
        "section.baseline_vs": "Línea base vs. ahora",
        "chart.time":          "tiempo (s)",
        "chart.y_pct":         "% de fuera de norma",
        "chart.legend.baseline":  "LÍNEA BASE (0%)",
        "chart.legend.now":       "AHORA",
        "chart.legend.watch":     "línea de vigilancia (±50%)",
        "chart.legend.oon":       "fuera de norma (±100%)",
        "chart.legend.episode":   "este episodio",
        "chart.spider.baseline":  "LÍNEA BASE (Zona Estable - Normal)",
        "chart.spider.now":       "AHORA (tu cara en este matiz)",
        # Spider radar axis labels — etiquetas cortas en lenguaje natural.
        "spider.left_brow_height_norm":  "CEJA IZQ",
        "spider.right_brow_height_norm": "CEJA DER",
        "spider.angle_LM":               "SONRISA IZQ",
        "spider.angle_RM":               "SONRISA DER",
        "spider.mouth_offset_norm":      "BOCA TORCIDA",
        "spider.side_left_norm":         "MEJILLA IZQ",
        "spider.side_right_norm":        "MEJILLA DER",
        "spider.yaw_proxy":              "GIRO CABEZA",
        "spider.pitch_proxy":            "CABECEO",
        "spider.roll_proxy":             "LADEO CABEZA",
        "spider.eye_mouth_ratio":        "CARA ALARGADA",
        "spider.diag_ratio":             "CARA TORCIDA",
        # Voice-command banner overlays.
        "voice.heard_start":             "ESCUCHÉ '{w}' — INICIANDO TOMA…",
        "voice.heard_stop":              "ESCUCHÉ '{w}' — DETENIENDO TOMA…",
        # Side / direction labels (ES).
        "side_label.left_brow_height_norm":  "◀ CEJA izq",
        "side_label.right_brow_height_norm": "CEJA der ▶",
        "side_label.angle_LM":               "◀ comisura izq",
        "side_label.angle_RM":               "comisura der ▶",
        "side_label.mouth_offset_norm":      "boca torcida",
        "side_label.mouth_line_norm":        "ancho de boca",
        "side_label.side_mouth_norm":        "ancho de boca",
        "side_label.side_left_norm":         "◀ lado izq",
        "side_label.side_right_norm":        "lado der ▶",
        "side_label.side_eye_norm":          "línea ojo / gafas",
        "side_label.eye_line_norm":          "línea ojo / gafas",
        "side_label.angle_LE":               "◀ rabillo ojo izq",
        "side_label.angle_RE":               "rabillo ojo der ▶",
        "side_label.yaw_proxy":              "◀ giro de cabeza ▶",
        "side_label.pitch_proxy":            "▲ mentón ▼",
        "side_label.roll_proxy":             "ladeo de cabeza",
        "side_label.diag_LE_RM_norm":        "diagonal LE → RM",
        "side_label.diag_RE_LM_norm":        "diagonal RE → LM",
        "side_label.diag_ratio":             "balance diagonal",
        "side_label.eye_mouth_ratio":        "proporción vertical",
        "side_label.parallelism_residual":   "no paralelo",
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
