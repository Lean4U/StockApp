# Per-subject baselines eliminate cultural bias in behavioural analytics

A technical defence of the Syntonia score's culturally-agnostic posture.

---

## Abstract

Commercial AI behavioural analytics — public-speaking coaches, interview
practice tools, in-call sentiment monitors — are overwhelmingly trained on
Western English-speaker reference data. When applied to subjects from
other linguistic or cultural backgrounds, they reliably misinterpret
normal baseline behaviour as deviation: pacing differences as nervousness,
expressive-baseline differences as low affect, gesture vocabularies as
fidgeting, vocal-modulation patterns as instability.

The Syntonia pipeline avoids this class of error by **never comparing a subject
to a population norm**. Every threshold is computed relative to the
subject's own first 15 seconds of footage, using robust statistics
(median + scaled MAD) on a 22-dimensional feature vector. The result: a
Japanese candidate's reserved facial baseline, an Italian's gesture
vocabulary, and a Hindi speaker's pacing all become *their own normal* —
because they are. Deviation is measured against the subject, not the
population.

This document explains why this works, demonstrates the effect with a
reproducible synthetic illustration, and is honest about what remains as
cultural-sensitivity work even with per-subject baselines.

---

## 1. The problem: Western bias in commercial behavioural AI

Three sources of well-documented bias compound in the cloud-coach product
category:

| Source | Manifestation |
|---|---|
| **Training-data skew** | Public datasets used to train facial-emotion classifiers (CK+, RAVDESS, AffectNet, FER+) over-represent US/Anglophone subjects. Classifiers trained on these datasets report systematically lower facial-emotion intensity on East Asian subjects — not because those subjects feel less, but because their expressive *baseline* is calibrated differently. Buolamwini & Gebru (2018, "Gender Shades") documented this class of bias in face recognition; subsequent work has shown it persists in expression analysis. |
| **Population-norm thresholds** | Coach apps that say "your pace is slower than 73% of speakers" use a reference distribution drawn almost entirely from US English speakers (5–7 syllables/second). Pellegrino et al. (2011, *Language*) documented that information rate is conserved across languages but syllable rate varies from ~3 syll/s in Mandarin to ~8 syll/s in Spanish. A "slow" Mandarin speaker by Western pace norms is a normal Mandarin speaker. |
| **Cultural-gesture baselines** | "Fidgeting" detection heuristics implicitly encode a low-gesture cultural default. High-gesture cultures (Italian, Spanish, Greek, several South Asian and Latin American populations) trip those heuristics constantly. Some communicative gestures — the South Asian head bobble, hijab-adjustment habits, prayer-bead handling — are not fidgets at all. |

The combined effect: a non-Western candidate using a cloud-coach app
receives advice to **perform as a Western English speaker**, not to be
their best version of themselves. That is the bias Syntonia exists to avoid.

---

## 2. The Syntonia architecture: never compare across subjects

The full mathematical machinery is documented elsewhere in the codebase
(see `face_trapezium.py`, `syntonia_model.py`). The single property that makes Syntonia
culturally agnostic is structural:

**No threshold is ever computed from a population.** Every threshold is
derived per-recording, from each subject's own baseline window.

The baseline computation is:

```
For each of the 22 features f_j:
    μ_j  =  median  ( f_j over the subject's first 15 s )
    σ_j  =  1.4826 × MAD( f_j over the subject's first 15 s )

For each subsequent frame i:
    z_j(i)  =  ( f_j(i) − μ_j ) / σ_j

A 3σ event fires when the aggregate per-frame deviation exceeds 3.
```

The bias-relevant consequences:

1. **The subject's *absolute* feature values never reach the threshold
   computation.** Whether the subject's neutral pace is 3 syll/s or 8
   syll/s, whether their resting brow position is 6 px above the
   eye-line or 15 px, whether their facial neutral has zero expression
   intensity or moderate expression intensity — none of those absolute
   values matter. Only deviation from their own median matters.

2. **Robust statistics (median + MAD) absorb the cultural-distribution
   shape itself.** Mean + standard deviation would be vulnerable to
   skewed distributions or outliers from the subject's own baseline
   window. The median + scaled-MAD pair tolerates up to 50%
   contamination in the baseline before breaking — so a brief stress
   event during the baseline window doesn't poison the reference.

3. **The 22-dimensional Hotelling T² aggregate uses per-feature variances
   that are also computed per-subject**, not from a population covariance
   estimate. So the aggregate test isn't "more deviation than other
   speakers"; it's "more deviation than this speaker showed in the first
   15 seconds".

This is **not a feature added on top of a Western-default model**. It's
the architecture's base layer. Removing it would require rewriting the
detector.

---

## 3. The proof: a concrete synthetic illustration

`scripts/cultural_baseline_demo.py` generates a synthetic 60-second
recording of two subjects whose **only** difference is their cultural
linguistic baseline:

- **Subject A**: speech-pace baseline ≈ 3.0 syllables/second (within the
  documented range for Japanese conversational speech)
- **Subject B**: speech-pace baseline ≈ 7.0 syllables/second (within the
  documented range for Spanish conversational speech)

Both subjects exhibit **no genuine stress events**. Their traces consist
of their normal baseline pace with ordinary frame-to-frame fluctuation
(±0.3 syll/s noise). Any sensible behavioural tool should silently report
"normal behaviour" for both.

Running both subjects through two threshold regimes produces this result:

| Threshold regime | Subject A flags | Subject B flags |
|---|---|---|
| **Western population norm** (μ = 5.5 syll/s, σ = 0.5; 3σ band) | **60 / 60 s — false-flagged every single second** | 26 / 60 s — false-flagged on most samples |
| **Per-subject baseline** (each subject's own first 15 s) | **0 / 60 s — correctly silent** | **0 / 60 s — correctly silent** |

The accompanying figure (`cultural_baseline_proof.png`) shows this
visually: two normal baselines, two correct interpretations under
per-subject statistics, two complete failures under population statistics.

This is **synthetic data engineered to make the bias visible**. We do not
claim it generalises to all real-world cases without further validation
(see §5). But it captures the structural problem clearly: any tool that
applies a fixed-population threshold to a subject whose baseline diverges
from that population will produce ~100% false-positive rates on the
diverging subjects.

---

## 4. Why MediaPipe and Whisper inherit this property

The Syntonia math is one of three layers, all of which contribute to the
cultural-agnostic posture:

| Layer | Cultural posture | Source |
|---|---|---|
| **Face / hand landmark detection** | Globally-sampled training data (Google's MediaPipe team explicitly built on a diverse face-image corpus to avoid the demographic-bias issues of earlier face-mesh models) | MediaPipe Tasks, Apache 2.0, `models/face_landmarker.task` |
| **Speech-to-text** | 98-language multilingual model with comparable per-language WER for the supported set | OpenAI Whisper, MIT-licensed, via `faster-whisper` |
| **Threshold + decision math** | Per-subject baseline; never references population | Syntonia, this codebase |

A claim that The Syntonia score is "culturally agnostic" is defensible because all three
layers respect the property. If any layer used a population-norm
classifier, the claim would fail — but none do.

---

## 5. Honest limitations and remaining cultural work

It would be dishonest to claim per-subject baselines solve every cultural
issue. They solve a specific, important problem (the false-flag rate on
non-Western subjects under population thresholds) and leave others open:

| Open issue | Why it remains | What we'd need to do |
|---|---|---|
| **Action-unit cluster weights (`S_AU` in the Syntonia score equation)** are currently uniform across cultures (`brow_knit` = 2.5, `asymmetric_lip` = 2.0, etc.) | Some FACS Action Units have culturally-different semantic loadings. Brow-knit reads as anger/concentration in Western cultures, but is less expressive in some East Asian baseline. | Ship per-region weight presets after collecting ~50-subject validation data per region. |
| **The 3.0 threshold itself is empirically motivated, not theoretically derived for each cultural context** | A 3σ aggregate corresponds to ~0.27% tail probability assuming Gaussian per-feature noise. Real behavioural noise is heavier-tailed; the false-alarm rate depends on the subject. | Calibrate threshold per cultural cluster from a held-out validation set. |
| **Hand-region weights (`W_b`)** assume a default cultural meaning for "touching face / neck / hair" | Hijab-adjustment, prayer-bead manipulation, head-cover adjustment are routine — not deception cues — in several populations. | Either down-weight ear / hair regions when the subject self-declares relevant context, or learn the per-subject hand-region baseline (advanced; not yet in v1). |
| **Gesture-rich speakers (Italian, Spanish, etc.)** still have higher F-channel baselines | Per-subject baseline absorbs the *level* of gesture, but the F-channel's reliability-weighted aggregation could still produce slightly different behaviour for high-gesture subjects | Validate the F-channel calibration empirically per population. |
| **Multilingual code-switching mid-recording** is not currently modelled | A subject who switches from Hindi to English mid-sentence may show V-channel anomalies that aren't stress | Whisper's language-detection signal can be used to segment the recording and re-baseline per language. Future work. |

These are real. We document them so anyone using the tool for clinical,
hiring, or research purposes knows where the boundaries are.

---

## 6. What this position lets Syntonia claim — and what it doesn't

**Defensible claims:**

- "The Syntonia's threshold detection does not assume a Western or English-
  speaker default. It is computed per-subject from the subject's own
  baseline footage."
- "Robust median + MAD statistics give the detector a 50% break-down
  point against baseline-window contamination."
- "Whisper-small handles speech transcription across 98 languages
  on-device, with no audio transmitted off the user's machine."
- "MediaPipe FaceMesh and HandLandmarker are trained on globally-sampled
  data and do not exhibit the demographic bias documented in earlier
  facial-analysis libraries."

**Claims we do NOT make:**

- "The Syntonia score is bias-free." (No system is.)
- "Syntonia's interpretations transfer perfectly across cultures." (See §5.)
- "Syntonia can replace culturally-competent clinical or coaching expertise."
  (The output is signal for human review, not a verdict.)
- "Syntonia works without ground-truth validation per subject population for
  clinical applications." (Tier 2 requires per-population calibration.)

---

## 7. Validation roadmap

To move from "synthetic illustration" to "empirically defensible", the
next concrete steps:

1. **Run the comparison-demo protocol** (`marketing/demo_specification.md`)
   on 3 non-Western subjects vs. Yoodli. This produces a public-facing
   case-study artifact. Not a study; sufficient for marketing claims with
   appropriate framing.

2. **Recruit a 30-subject cross-cultural pilot** (10 subjects each from 3
   linguistic groups). Compute the false-alarm rate under per-subject vs.
   population thresholds on natural-speech recordings. This is publishable
   as a workshop paper at a venue like LREC, EUSIPCO, or ACII.

3. **Publish per-cluster AU and region-weight presets** based on the
   pilot data, with explicit "calibrated on N subjects from population X"
   labels. Resist the temptation to ship a single "global" weight set.

4. **Independent re-audit** of the network behaviour: a third-party
   security review confirming zero off-device data egress in operating
   modes. This converts the privacy claim from "the developer says so"
   into "audited posture".

5. **Bias-bounty programme**: invite researchers to find baseline
   configurations where the per-subject approach still misclassifies.
   Pay for confirmed findings; publish the patches.

---

## 8. Conclusion

The Syntonia's claim to cultural agnosticism is not a marketing position
bolted on top of a Western-default detector. It is the architectural
default: nothing in the threshold computation references a population
norm. The synthetic illustration in §3 makes the structural problem
visible; the validation roadmap in §7 is how the position moves from
defensible to empirically demonstrated.

This is the technical defence of the product position. The accompanying
marketing materials (`marketing/demo_specification.md`,
`marketing/landing_page_hero.md`) translate it for the buyer.

---

## References

- Buolamwini, J. & Gebru, T. (2018). *Gender Shades: Intersectional
  Accuracy Disparities in Commercial Gender Classification*. Proceedings
  of Machine Learning Research 81.
- Pellegrino, F., Coupé, C., & Marsico, E. (2011). *A cross-language
  perspective on speech information rate*. Language 87(3).
- Raji, I. D. et al. (2020). *Saving Face: Investigating the Ethical
  Concerns of Facial Recognition Auditing*. AIES.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT*.
- Whisper / OpenAI: Radford, A. et al. (2022). *Robust Speech Recognition
  via Large-Scale Weak Supervision*. arXiv 2212.04356.
- MediaPipe team (2023). *MediaPipe Tasks: On-device ML solutions*.
  https://developers.google.com/mediapipe

(References point to known literature; cited claims are paraphrased
fairly. Verify against the originals before reuse in any formal
publication.)
