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


def explain(feature: str) -> FeatureExplanation:
    """Return the FeatureExplanation for a feature name, with a generic
    fallback so unknown features never crash the dashboard."""
    return _GLOSSARY.get(
        feature,
        FeatureExplanation(
            short=feature,
            physical=f"Internal feature: {feature}.",
            candidates=["(no plain-language explanation registered)"],
            socratic="Notice this moment — what were you doing?",
        ),
    )


def category(feature: str) -> str:
    """Coarse bucket for natural-language summaries."""
    if feature in BEHAVIOURAL_FEATURES:
        return "behavioural"
    if feature in HARDWARE_FEATURES:
        return "hardware"
    if feature in POSE_FEATURES:
        return "pose"
    return "other"
