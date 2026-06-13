"""Plain-language explanations for every trapezium feature.

These power the natural-language narration in dashboard_live.py.

Per the brand-architecture framework (marketing/brand_architecture.md):

  * Principle 2 — Self-discovery, not instruction.
    We describe what the GEOMETRY did, not what the subject "felt".
  * Principle 4 — Socratic method.
    We end with a question, not a conclusion.
  * Principle 5 — Validated by science, in plain language.
    Where a published citation exists, we name it.
  * Principle 6 — No deterministic claims from math alone.
    "Consistent with", "compatible with" — never "this means you did X".
  * Principle 7 — No bias insertion.
    We list two or more candidate physical interpretations when the same
    feature can be produced by multiple movements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FeatureExplanation:
    short: str            # one-line, used in lists
    physical: str         # what the feature actually measures (geometric)
    candidates: List[str] # plausible physical movements that produce a deviation
    citation: Optional[str] = None  # named published source if applicable
    science_summary: Optional[str] = None  # plain-language one-liner of what the science says
    socratic: str = ""    # the question the dashboard puts to the subject


_GLOSSARY: Dict[str, FeatureExplanation] = {
    "eye_line_norm": FeatureExplanation(
        short="Eye-line length (normalized)",
        physical=(
            "Horizontal distance between your outer left and right eye corners, "
            "scaled by overall face size."
        ),
        candidates=[
            "Head tilt toward / away from the camera",
            "Squinting (eye-corners pulled toward the nose)",
            "Glasses sitting in a different position",
        ],
        socratic="Did you lean toward or away from the screen at this moment?",
    ),
    "mouth_line_norm": FeatureExplanation(
        short="Mouth-line length (normalized)",
        physical="Horizontal distance between your left and right mouth corners.",
        candidates=[
            "Wide open mouth (laugh, surprise, broad vowel)",
            "Pursed lips (rounded vowel, hesitation)",
            "Asymmetric expression pulling one corner laterally",
        ],
        socratic="What were you saying — and how wide did your mouth open to say it?",
    ),
    "side_left_norm": FeatureExplanation(
        short="Left trapezium side (left eye → left mouth)",
        physical=(
            "Vertical distance between your left outer-eye corner and your "
            "left mouth corner, normalized by face scale."
        ),
        candidates=[
            "Head turn to the right (left side stretches in projection)",
            "Left cheek raise (smile pulling the left corner up)",
            "Jaw drop on the left side",
        ],
        socratic="Was your weight shifting in the chair?",
    ),
    "side_right_norm": FeatureExplanation(
        short="Right trapezium side (right eye → right mouth)",
        physical=(
            "Vertical distance between your right outer-eye corner and your "
            "right mouth corner, normalized by face scale."
        ),
        candidates=[
            "Head turn to the left (right side stretches in projection)",
            "Right cheek raise (smile pulling the right corner up)",
            "Right-sided jaw drop",
        ],
        socratic="Were you turning toward something on screen?",
    ),
    "side_eye_norm": FeatureExplanation(
        short="Eye-side compression",
        physical=(
            "The eye-line length relative to the rest of the trapezium. "
            "Highly sensitive to the position of glasses frames on the bridge "
            "of the nose."
        ),
        candidates=[
            "Glasses settled into a different position",
            "Distance from the webcam shifted",
            "Persistent head tilt toward / away from camera",
        ],
        socratic="Are your glasses sitting differently than when you enrolled?",
    ),
    "side_mouth_norm": FeatureExplanation(
        short="Mouth-side width",
        physical=(
            "Mouth-line length relative to the rest of the trapezium. "
            "Captures lateral mouth opening, which differs systematically "
            "between languages with more open vowels (Spanish /a/, /e/, /o/) "
            "vs more rounded ones (English back vowels)."
        ),
        candidates=[
            "Speaking a language with wider open vowels",
            "Wide smile",
            "Wide open mouth (laugh, broad surprise)",
        ],
        socratic="Were you switching languages or producing wide-open vowels?",
    ),
    "angle_LE": FeatureExplanation(
        short="Left-eye interior angle",
        physical=(
            "Interior angle of the trapezium at the left outer-eye corner."
        ),
        candidates=[
            "Head pitch (chin up / down) reprojecting the left side",
            "Left brow furrow or squint",
            "Glasses-frame edge interfering with the landmark",
        ],
        socratic="Did your head tilt slightly here?",
    ),
    "angle_RE": FeatureExplanation(
        short="Right-eye interior angle",
        physical=(
            "Interior angle of the trapezium at the right outer-eye corner."
        ),
        candidates=[
            "Head pitch (chin up / down) reprojecting the right side",
            "Right brow furrow or squint",
            "Glasses-frame edge interfering with the landmark",
        ],
        socratic="Did your head tilt slightly here?",
    ),
    "angle_LM": FeatureExplanation(
        short="Left-mouth interior angle",
        physical="Interior angle of the trapezium at the left mouth corner.",
        candidates=[
            "Smile pulling the left corner up and back",
            "Asymmetric mouth movement (smirk, half-smile)",
            "Left-sided lip compression",
        ],
        citation="Delor et al. 2021 — facial asymmetry predicts genuineness "
                 "judgements (Frontiers in Psychology 12, 727446).",
        science_summary=(
            "When one corner of a smile moves more than the other, the smile "
            "reads as genuine. Perfectly symmetric smiles often read as "
            "performative or forced."
        ),
        socratic=(
            "Was the expression here lopsided — one side moving more than the other?"
        ),
    ),
    "angle_RM": FeatureExplanation(
        short="Right-mouth interior angle",
        physical="Interior angle of the trapezium at the right mouth corner.",
        candidates=[
            "Smile pulling the right corner up and back",
            "Asymmetric mouth movement (smirk, half-smile)",
            "Speaking specific phonemes that pull the right corner",
        ],
        citation="Delor et al. 2021 — facial asymmetry predicts genuineness "
                 "judgements (Frontiers in Psychology 12, 727446).",
        science_summary=(
            "When one corner of a smile moves more than the other, the smile "
            "reads as genuine. Perfectly symmetric smiles often read as "
            "performative or forced."
        ),
        socratic="Did one side of your mouth move more than the other?",
    ),
    "diag_LE_RM_norm": FeatureExplanation(
        short="Trapezium diagonal LE → RM",
        physical=(
            "Length of the trapezium diagonal from the left outer-eye corner "
            "to the right mouth corner."
        ),
        candidates=[
            "Head yaw (turning right)",
            "Diagonal facial asymmetry — left brow up + right mouth down, or "
            "vice versa",
            "Combined chin-up + side-look",
        ],
        socratic="Did you glance to one side?",
    ),
    "diag_RE_LM_norm": FeatureExplanation(
        short="Trapezium diagonal RE → LM",
        physical=(
            "Length of the trapezium diagonal from the right outer-eye corner "
            "to the left mouth corner."
        ),
        candidates=[
            "Head yaw (turning left)",
            "Diagonal facial asymmetry",
            "Combined chin-up + side-look",
        ],
        socratic="Did you glance to one side?",
    ),
    "diag_ratio": FeatureExplanation(
        short="Diagonal asymmetry ratio",
        physical=(
            "Ratio of the two trapezium diagonals. A perfectly symmetric, "
            "front-facing face has ratio near 1.0. Any departure means the "
            "face is no longer projecting symmetrically into the camera."
        ),
        candidates=[
            "Head yaw (turning toward one side)",
            "Asymmetric facial expression (one cheek lifting more than the other)",
            "Camera angle changed",
        ],
        citation="Delor et al. 2021 (asymmetry → genuineness).",
        science_summary=(
            "Diagonal asymmetry of the face shifts in micro-seconds when "
            "emotion is genuine. Forced expressions tend to keep the "
            "diagonals balanced."
        ),
        socratic="Was your head pointed straight at the camera here?",
    ),
    "eye_mouth_ratio": FeatureExplanation(
        short="Eye-line / mouth-line proportion",
        physical=(
            "Vertical proportion between the eye line and the mouth line. "
            "Changes when the head pitches up or down, or when the mouth "
            "opens wider."
        ),
        candidates=[
            "Chin tilted up or down (reading text on a screen)",
            "Mouth opened wider (open vowel, surprise, yawn)",
            "Subject leaned forward or back",
        ],
        socratic="Were you reading off the screen, looking down or up?",
    ),
    "parallelism_residual": FeatureExplanation(
        short="Eye-line / mouth-line parallelism residual",
        physical=(
            "How non-parallel the eye line and mouth line are. Captures the "
            "head pitch independent of distance to camera."
        ),
        candidates=[
            "Head pitched up or down",
            "Jaw drop (lower lip moving down independently of head)",
            "Brow tilt independent of mouth",
        ],
        socratic="Were you nodding here, or looking at a different part of the screen?",
    ),
    "left_brow_height_norm": FeatureExplanation(
        short="Left brow elevation",
        physical=(
            "Vertical distance from the left brow point to the left outer-eye "
            "corner."
        ),
        candidates=[
            "Surprise / brow raise (FACS AU 1 + AU 2)",
            "Concentration furrow if it goes downward instead",
            "Asymmetric brow raise (only one side)",
        ],
        citation="Ekman & Friesen — FACS AU 1 (Inner Brow Raiser) and AU 2 "
                 "(Outer Brow Raiser).",
        science_summary=(
            "The brow flash — both brows up for about 200 ms — is a "
            "cross-cultural signal of attention or recognition. A one-sided "
            "raise is more often a question, a check-in or scepticism."
        ),
        socratic="Were you surprised, concentrating, or asking a question?",
    ),
    "right_brow_height_norm": FeatureExplanation(
        short="Right brow elevation",
        physical=(
            "Vertical distance from the right brow point to the right "
            "outer-eye corner."
        ),
        candidates=[
            "Surprise / brow raise (FACS AU 1 + AU 2)",
            "Concentration furrow if it goes downward",
            "Asymmetric brow raise (only one side)",
        ],
        citation="Ekman & Friesen — FACS AU 1 / AU 2.",
        science_summary=(
            "The brow flash — both brows up for about 200 ms — is a "
            "cross-cultural signal of attention or recognition. A one-sided "
            "raise is more often a question, a check-in or scepticism."
        ),
        socratic="Were you surprised, concentrating, or asking a question?",
    ),
    "yaw_proxy": FeatureExplanation(
        short="Head yaw (left ↔ right turn)",
        physical=(
            "Geometric proxy for head rotation around the vertical axis, "
            "derived from the relative widths of the two trapezium sides."
        ),
        candidates=[
            "Looking left or right at the screen",
            "Turning toward someone speaking to you",
            "Reading across a wide passage",
        ],
        socratic="Were you scanning your screen here?",
    ),
    "pitch_proxy": FeatureExplanation(
        short="Head pitch (chin up ↔ down)",
        physical=(
            "Geometric proxy for head tilt around the horizontal axis, "
            "derived from the eye-mouth-ratio departure from the median."
        ),
        candidates=[
            "Looking down at text or notes",
            "Chin lift while thinking",
            "Subject's chair height changed",
        ],
        socratic="Were you looking down at a paragraph, or up at the camera?",
    ),
    "roll_proxy": FeatureExplanation(
        short="Head roll (side-tilt)",
        physical=(
            "Geometric proxy for head tilt around the forward axis (ear "
            "toward shoulder)."
        ),
        candidates=[
            "Casual head tilt (curiosity, listening)",
            "Posture drift",
            "Subject leaned on one elbow",
        ],
        socratic="Were you tilting your head to one side here?",
    ),
    "mouth_offset_norm": FeatureExplanation(
        short="Mouth horizontal offset from trapezium centroid",
        physical=(
            "How far the mouth midpoint is shifted left or right relative to "
            "the centre of the eye-to-mouth quadrilateral."
        ),
        candidates=[
            "Asymmetric smile or smirk",
            "One-sided lip movement (speaking specific phonemes)",
            "Half-grimace / lip purse",
        ],
        citation="Delor et al. 2021 — asymmetric mouth movements are the "
                 "single most reliable visible marker of forced vs spontaneous "
                 "smile.",
        science_summary=(
            "Mouth offset from the face midline is the strongest single "
            "indicator of whether a smile is spontaneous: spontaneous smiles "
            "are visibly lopsided, forced smiles are not."
        ),
        socratic="Did the expression pull more to one side than the other?",
    ),
}


# Higher-level categorization for the session summary.
BEHAVIOURAL_FEATURES = {
    "left_brow_height_norm", "right_brow_height_norm",
    "angle_LM", "angle_RM",
    "mouth_offset_norm",
}

HARDWARE_FEATURES = {
    "side_eye_norm", "angle_LE", "angle_RE",
    "eye_line_norm",
    "diag_LE_RM_norm", "diag_RE_LM_norm", "diag_ratio",
}

POSE_FEATURES = {
    "yaw_proxy", "pitch_proxy", "roll_proxy",
    "parallelism_residual", "eye_mouth_ratio",
    "side_left_norm", "side_right_norm",
}


def explain(feature: str, lang: str = "en") -> FeatureExplanation:
    """Return the FeatureExplanation for a feature name, in the user's
    output language when available. Falls back to English per-field if
    a translation isn't registered yet.
    """
    en_entry = _GLOSSARY.get(
        feature,
        FeatureExplanation(
            short=feature,
            physical=f"Internal feature: {feature}.",
            candidates=["(no plain-language explanation registered)"],
            socratic="Notice this moment — what were you doing?",
        ),
    )
    if lang == "en" or lang not in _LOCALIZED_GLOSSARY:
        return en_entry
    loc = _LOCALIZED_GLOSSARY[lang].get(feature)
    if loc is None:
        return en_entry
    # Merge: prefer localized fields, fall back to English per-field.
    return FeatureExplanation(
        short=loc.get("short", en_entry.short),
        physical=loc.get("physical", en_entry.physical),
        candidates=loc.get("candidates", en_entry.candidates),
        citation=loc.get("citation", en_entry.citation),
        science_summary=loc.get("science_summary", en_entry.science_summary),
        socratic=loc.get("socratic", en_entry.socratic),
    )


# Localized field overrides per language. Each entry is feature_name →
# dict of optional field overrides. Unset fields fall back to English.
_LOCALIZED_GLOSSARY: Dict[str, Dict[str, dict]] = {
    "es": {
        "left_brow_height_norm": {
            "short": "Elevación de ceja izquierda",
            "physical": "Distancia vertical desde el punto de la ceja izquierda hasta el rabillo exterior del ojo izquierdo.",
            "candidates": [
                "Sorpresa / levantamiento de cejas (FACS AU 1 + AU 2)",
                "Fruncimiento de concentración si va hacia abajo",
                "Levantamiento asimétrico (solo un lado)",
            ],
            "citation": "Ekman y Friesen — FACS AU 1 (Elevador interno de ceja) y AU 2 (Elevador externo de ceja).",
            "science_summary": "El destello de cejas — ambas cejas arriba durante unos 200 ms — es una señal transcultural de atención o reconocimiento. Un levantamiento unilateral suele ser una pregunta, una verificación o escepticismo.",
            "socratic": "¿Estabas sorprendido, concentrándote o formulando una pregunta?",
        },
        "right_brow_height_norm": {
            "short": "Elevación de ceja derecha",
            "physical": "Distancia vertical desde el punto de la ceja derecha hasta el rabillo exterior del ojo derecho.",
            "candidates": [
                "Sorpresa / levantamiento de cejas (FACS AU 1 + AU 2)",
                "Fruncimiento de concentración si va hacia abajo",
                "Levantamiento asimétrico (solo un lado)",
            ],
            "citation": "Ekman y Friesen — FACS AU 1 / AU 2.",
            "science_summary": "El destello de cejas — ambas cejas arriba durante unos 200 ms — es una señal transcultural de atención o reconocimiento. Un levantamiento unilateral suele ser una pregunta, una verificación o escepticismo.",
            "socratic": "¿Estabas sorprendido, concentrándote o formulando una pregunta?",
        },
        "angle_LM": {
            "short": "Ángulo interior de la comisura izquierda",
            "physical": "Ángulo interior del trapecio en la comisura izquierda de la boca.",
            "candidates": [
                "Sonrisa que tira la comisura izquierda hacia arriba y atrás",
                "Movimiento bucal asimétrico (sonrisa torcida, media sonrisa)",
                "Compresión del labio izquierdo",
            ],
            "citation": "Delor et al. 2021 — la asimetría facial predice los juicios de autenticidad (Frontiers in Psychology 12, 727446).",
            "science_summary": "Cuando una comisura de la sonrisa se mueve más que la otra, la sonrisa parece genuina. Las sonrisas perfectamente simétricas a menudo parecen actuadas o forzadas.",
            "socratic": "¿Fue la expresión asimétrica aquí — un lado moviéndose más que el otro?",
        },
        "angle_RM": {
            "short": "Ángulo interior de la comisura derecha",
            "physical": "Ángulo interior del trapecio en la comisura derecha de la boca.",
            "candidates": [
                "Sonrisa que tira la comisura derecha hacia arriba y atrás",
                "Movimiento bucal asimétrico (sonrisa torcida, media sonrisa)",
                "Habla de fonemas específicos que tiran de la comisura derecha",
            ],
            "citation": "Delor et al. 2021 — la asimetría facial predice los juicios de autenticidad (Frontiers in Psychology 12, 727446).",
            "science_summary": "Cuando una comisura de la sonrisa se mueve más que la otra, la sonrisa parece genuina. Las sonrisas perfectamente simétricas a menudo parecen actuadas o forzadas.",
            "socratic": "¿Movió un lado de tu boca más que el otro?",
        },
        "mouth_offset_norm": {
            "short": "Desplazamiento horizontal de la boca desde el centroide",
            "physical": "Cuánto se ha desplazado el punto medio de la boca a la izquierda o derecha respecto al centro del cuadrilátero ojo-boca.",
            "candidates": [
                "Sonrisa o mueca asimétrica",
                "Movimiento labial unilateral (al pronunciar fonemas específicos)",
                "Media mueca / labios fruncidos",
            ],
            "citation": "Delor et al. 2021 — los movimientos asimétricos de la boca son el único indicador visible más confiable de sonrisa forzada vs espontánea.",
            "science_summary": "El desplazamiento de la boca desde la línea media facial es el indicador único más fuerte de si una sonrisa es espontánea: las sonrisas espontáneas son visiblemente asimétricas, las forzadas no.",
            "socratic": "¿Tiró la expresión más hacia un lado que hacia el otro?",
        },
        "side_mouth_norm": {
            "short": "Anchura de la boca",
            "physical": "Longitud de la línea de la boca relativa al resto del trapecio. Captura la apertura lateral, que difiere sistemáticamente entre idiomas con vocales más abiertas (español /a/, /e/, /o/) y los más redondeados (vocales posteriores del inglés).",
            "candidates": [
                "Hablar un idioma con vocales abiertas más amplias",
                "Sonrisa amplia",
                "Boca muy abierta (risa, sorpresa amplia)",
            ],
            "socratic": "¿Estabas cambiando de idioma o produciendo vocales bien abiertas?",
        },
        "side_left_norm": {
            "short": "Lado izquierdo del trapecio (ojo izq → boca izq)",
            "physical": "Distancia vertical entre el rabillo exterior del ojo izquierdo y la comisura izquierda de la boca, normalizada por la escala del rostro.",
            "candidates": [
                "Giro de cabeza a la derecha (el lado izquierdo se estira en proyección)",
                "Elevación del pómulo izquierdo (sonrisa que tira hacia arriba)",
                "Caída de mandíbula del lado izquierdo",
            ],
            "socratic": "¿Estabas cambiando el peso en la silla?",
        },
        "side_right_norm": {
            "short": "Lado derecho del trapecio (ojo der → boca der)",
            "physical": "Distancia vertical entre el rabillo exterior del ojo derecho y la comisura derecha de la boca, normalizada por la escala del rostro.",
            "candidates": [
                "Giro de cabeza a la izquierda (el lado derecho se estira en proyección)",
                "Elevación del pómulo derecho (sonrisa que tira hacia arriba)",
                "Caída de mandíbula del lado derecho",
            ],
            "socratic": "¿Estabas girándote hacia algo en la pantalla?",
        },
        "yaw_proxy": {
            "short": "Giro de cabeza (izquierda ↔ derecha)",
            "physical": "Proxy geométrico para la rotación de cabeza alrededor del eje vertical, derivado de los anchos relativos de los dos lados del trapecio.",
            "candidates": [
                "Mirar a la izquierda o derecha en la pantalla",
                "Girarte hacia alguien que te habla",
                "Leer a través de un pasaje ancho",
            ],
            "socratic": "¿Estabas escaneando tu pantalla aquí?",
        },
        "pitch_proxy": {
            "short": "Inclinación de cabeza (mentón arriba ↔ abajo)",
            "physical": "Proxy geométrico para la inclinación de cabeza alrededor del eje horizontal, derivado de la desviación de la relación ojo-boca respecto a la mediana.",
            "candidates": [
                "Mirar hacia abajo a texto o notas",
                "Levantar el mentón mientras se piensa",
                "Cambió la altura de la silla del sujeto",
            ],
            "socratic": "¿Estabas mirando hacia abajo a un párrafo, o hacia arriba a la cámara?",
        },
        "roll_proxy": {
            "short": "Inclinación lateral de la cabeza",
            "physical": "Proxy geométrico para la inclinación de cabeza alrededor del eje frontal (oreja hacia hombro).",
            "candidates": [
                "Inclinación casual de cabeza (curiosidad, escuchando)",
                "Deriva postural",
                "El sujeto se apoyó en un codo",
            ],
            "socratic": "¿Estabas inclinando la cabeza hacia un lado aquí?",
        },
        "eye_mouth_ratio": {
            "short": "Proporción línea-ojo / línea-boca",
            "physical": "Proporción vertical entre la línea del ojo y la línea de la boca. Cambia cuando la cabeza se inclina arriba o abajo, o cuando la boca se abre más.",
            "candidates": [
                "Mentón inclinado arriba o abajo (leyendo texto en pantalla)",
                "Boca abierta más amplia (vocal abierta, sorpresa, bostezo)",
                "El sujeto se inclinó hacia adelante o atrás",
            ],
            "socratic": "¿Estabas leyendo de la pantalla, mirando abajo o arriba?",
        },
        "diag_ratio": {
            "short": "Proporción de asimetría diagonal",
            "physical": "Proporción de las dos diagonales del trapecio. Un rostro perfectamente simétrico mirando de frente tiene proporción cercana a 1.0. Cualquier desviación significa que el rostro ya no se proyecta simétricamente en la cámara.",
            "candidates": [
                "Giro de cabeza (girando hacia un lado)",
                "Expresión facial asimétrica (una mejilla se eleva más que la otra)",
                "Cambió el ángulo de la cámara",
            ],
            "citation": "Delor et al. 2021 (asimetría → autenticidad).",
            "science_summary": "La asimetría diagonal de la cara cambia en microsegundos cuando la emoción es genuina. Las expresiones forzadas tienden a mantener las diagonales balanceadas.",
            "socratic": "¿Estaba tu cabeza apuntando directamente a la cámara aquí?",
        },
        "parallelism_residual": {
            "short": "Residual de paralelismo ojo-boca",
            "physical": "Cuán no paralelas son la línea de los ojos y la línea de la boca. Captura la inclinación de cabeza independientemente de la distancia a la cámara.",
            "candidates": [
                "Cabeza inclinada arriba o abajo",
                "Caída de mandíbula (labio inferior moviéndose abajo independientemente de la cabeza)",
                "Inclinación de cejas independiente de la boca",
            ],
            "socratic": "¿Estabas asintiendo aquí, o mirando a otra parte de la pantalla?",
        },
        "angle_LE": {
            "short": "Ángulo interior del ojo izquierdo",
            "physical": "Ángulo interior del trapecio en el rabillo exterior del ojo izquierdo.",
            "candidates": [
                "Inclinación de cabeza (mentón arriba/abajo) reproyectando el lado izquierdo",
                "Fruncimiento o entrecierre izquierdo",
                "Borde de las gafas interfiriendo con el punto de referencia",
            ],
            "socratic": "¿Inclinaste ligeramente la cabeza aquí?",
        },
        "angle_RE": {
            "short": "Ángulo interior del ojo derecho",
            "physical": "Ángulo interior del trapecio en el rabillo exterior del ojo derecho.",
            "candidates": [
                "Inclinación de cabeza (mentón arriba/abajo) reproyectando el lado derecho",
                "Fruncimiento o entrecierre derecho",
                "Borde de las gafas interfiriendo con el punto de referencia",
            ],
            "socratic": "¿Inclinaste ligeramente la cabeza aquí?",
        },
        "side_eye_norm": {
            "short": "Compresión del lado del ojo",
            "physical": "La longitud de la línea del ojo relativa al resto del trapecio. Muy sensible a la posición de las monturas de gafas sobre el puente de la nariz.",
            "candidates": [
                "Las gafas se asentaron en una posición diferente",
                "Cambió la distancia a la webcam",
                "Inclinación persistente de cabeza hacia/desde la cámara",
            ],
            "socratic": "¿Están tus gafas en una posición diferente que cuando hiciste la inscripción?",
        },
        "eye_line_norm": {
            "short": "Longitud de la línea del ojo (normalizada)",
            "physical": "Distancia horizontal entre los rabillos exteriores de los ojos izquierdo y derecho, escalada por el tamaño total de la cara.",
            "candidates": [
                "Inclinación de cabeza hacia/desde la cámara",
                "Entrecerrar los ojos (rabillos jalados hacia la nariz)",
                "Las gafas están en una posición diferente",
            ],
            "socratic": "¿Te inclinaste hacia o lejos de la pantalla en este momento?",
        },
        "mouth_line_norm": {
            "short": "Longitud de la línea de la boca (normalizada)",
            "physical": "Distancia horizontal entre las comisuras izquierda y derecha de la boca.",
            "candidates": [
                "Boca bien abierta (risa, sorpresa, vocal amplia)",
                "Labios fruncidos (vocal redondeada, vacilación)",
                "Expresión asimétrica que jala una comisura lateralmente",
            ],
            "socratic": "¿Qué estabas diciendo — y cuán amplia abriste la boca para decirlo?",
        },
        "diag_LE_RM_norm": {
            "short": "Diagonal del trapecio LE → RM",
            "physical": "Longitud de la diagonal del trapecio desde el rabillo exterior del ojo izquierdo a la comisura derecha de la boca.",
            "candidates": [
                "Giro de cabeza (girando a la derecha)",
                "Asimetría facial diagonal — ceja izquierda arriba + boca derecha abajo, o viceversa",
                "Mentón arriba + mirada lateral combinada",
            ],
            "socratic": "¿Echaste un vistazo a un lado?",
        },
        "diag_RE_LM_norm": {
            "short": "Diagonal del trapecio RE → LM",
            "physical": "Longitud de la diagonal del trapecio desde el rabillo exterior del ojo derecho a la comisura izquierda de la boca.",
            "candidates": [
                "Giro de cabeza (girando a la izquierda)",
                "Asimetría facial diagonal",
                "Mentón arriba + mirada lateral combinada",
            ],
            "socratic": "¿Echaste un vistazo a un lado?",
        },
    },
}


def category(feature: str) -> str:
    """Coarse bucket for natural-language summaries."""
    if feature in BEHAVIOURAL_FEATURES:
        return "behavioural"
    if feature in HARDWARE_FEATURES:
        return "hardware"
    if feature in POSE_FEATURES:
        return "pose"
    return "other"
