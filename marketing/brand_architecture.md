# SyntoniaPro — Brand Architecture

Single canonical reference for every brand decision locked through May 2026.
Whenever a downstream artifact (whitepaper, code, marketing copy, sales
deck, legal filing, partnership outreach) needs to know "what's our name,
what's our positioning, what's our framework", the answer lives here.

---

## 1. Layered naming

| Layer | Name | Status |
|---|---|---|
| **Consumer brand** | **SyntoniaPro** | ✅ Locked. Primary domain `SyntoniaPro.com` owned. |
| **Technical / mathematical engine** | **Syntonia Model** | ✅ Locked. Internal name for the measurement framework. |
| **Equation** | `Syntonia(t) = α·V(t) + β·F(t) + γ·M(t)` with `α + β + γ = 1`, threshold 3.0 over rolling 5-second window | ✅ Locked. The equation document remains the historical source. |
| **Subindex channels** | **V** (Voice) · **F** (Fidget) · **M** (Micro-expressions / face) | ✅ Locked. |
| **Engineering codename** *(internal docs only)* | `Syntonia` (one word, no `Pro`) | For code module names, function prefixes, variable names — shorter for ergonomic typing |

### Trademark posture

- **SyntoniaPro** searched at USPTO TESS Classes 9 + 42, WIPO Global Brand Database, EUIPO TMview: **no LIVE conflicts** as of May 2026.
- Defensive domain bundle: registered `SyntoniaPro.com`; additional variants under active acquisition (`syntoniapro.app`, `syntonia.coach`, etc.).
- **Syntonia Model** is treated as an internal technical descriptor, not a separately-registered trademark — same pattern as TensorFlow's "Eager Mode", Adobe's "Creative Cloud Apps", etc.
- Optional next step: USPTO 1B Intent-to-Use application for `SYNTONIAPRO` in Class 9 + 42 (~$700 total). Reserves priority date before product launch.

---

## 2. Positioning statement

> **For ambitious professionals anywhere in the world preparing for
> high-stakes communication — in your first language or your third, in
> your home country or a job market 7,000 miles away — SyntoniaPro is
> the first behavioural-friction coach that calibrates to YOUR cultural
> and linguistic baseline, not a Western-English default.**
>
> No cloud. No subscription. No data leaves your device.

### One-line elevator pitch

*SyntoniaPro is a private practice partner that helps you align with your best
self before high-stakes moments — privately, in your own language, against
your own baseline.*

### Tagline candidates (pick one)

- "Calibrated to you, not to the average."
- "Your private practice partner."
- "Privately. In your language. Against your own baseline."
- "The science of being your best self."

---

## 3. Guiding principles (the framework)

These eleven principles are the constitutional document. Every product
decision, every feature, every marketing claim is tested against them.
Anything that violates a principle is reworked before it ships.

| # | Principle | Operational meaning |
|---|---|---|
| 1 | **Defined persona, real problem** | Every feature starts from a real subject doing a real thing. No solution-in-search-of-problem features. |
| 2 | **Self-discovery, not instruction** | The product reveals; the subject concludes. No "the AI says you should…" framing. |
| 3 | **Subject in control, always** | No nudges, no recommendations, no decisions made for the subject. |
| 4 | **Socratic method** | Provocations and questions, not assertions. The product asks; the subject answers. |
| 5 | **Validated by science, in plain language** | Every claim links to published research; everyday wording; deeper sources on demand. |
| 6 | **No deterministic claims from math alone** | Numbers without research backing are interesting, not authoritative. |
| 7 | **No bias insertion, no guided thinking** | Frame neutrally; let the subject draw the conclusion. |
| 8 | **Real-time engagement assessment** | The product monitors whether the subject is engaged and adapts. |
| 9 | **Builds on prior interactions** | Continuity across sessions; the relationship deepens. |
| 10 | **Genuinely supportive** | No false praise, no toxic positivity, no performative encouragement. |
| 11 | **Subject-product alignment at every step** | The product checks in. It doesn't push. |

---

## 4. Persona — Priya (primary)

> Priya is a senior software engineer in Bangalore preparing for
> staff-engineer interviews at Google's Zurich office. The interview
> will be in **English — her third language after Tamil and Hindi.** She
> needs a tool that:
>
> 1. Doesn't penalise her L2/L3 pacing as a Western coach would
> 2. Doesn't read her reserved facial baseline as low affect
> 3. Doesn't put her practice footage on a US cloud her current
>    employer's DLP would notice
>
> She wants a coach that **records her in Bangalore, learns her baseline
> in English, tells her where she actually wobbled — and never leaves
> her laptop.**

### Persona variations across the launch markets

| Persona | Country | Languages | Anchor concern |
|---|---|---|---|
| **Priya** | India | Tamil / Hindi + EN | Western pace norms penalise L2/L3 speakers |
| **Kenji** | Japan | Japanese + EN | Lower facial-expressiveness baseline + cannot tolerate Western-pop-norm scoring |
| **Marco** | Italy | Italian + EN + FR | Natural gesture vocabulary reads as "fidget" to Western tools |
| **Aisha** | Egypt | Arabic + EN | Hijab framing + L2 English; intense privacy concerns |
| **Lucas** | Brazil | Portuguese + EN | Wants to track confidence across languages |
| **Wei** | China | Mandarin + EN | Lower-intensity facial baseline; cannot use US-cloud tools |
| **Olu** | Nigeria | Yoruba + English | Wants to identify code-switching moments |

Each persona's concerns are absorbed by the same per-subject baseline
architecture; no per-persona customisation needed.

---

## 5. Target tiers (use-case map)

| Tier | Use case | Status | Hardware target |
|---|---|---|---|
| **1 — Self-coaching** | Subject records themselves; reviews their own Syntonia events | **Primary launch focus.** v1 product. | Mac mini / Beelink / Windows laptop |
| **2 — Clinical / coaching pair** | Subject + clinician co-review sessions | Deferred to year 2; speech-language pathology is cleanest first medical adjacency | Per-clinician workstation |
| **3 — Research / academic** | Multimodal behavioural research | Deferred; needs IRB pathway | Per-researcher workstation |
| **4 — Sales / customer-experience** | Reps review their own calls | Deferred to year 2 | Mini-PC per rep |
| **5 — Adversarial / surveillance** | Hiring screening, interrogation aid, covert workplace monitoring | **Do not ship.** Documented as explicitly out-of-scope. |

---

## 6. Scientific anchoring (citation lineage)

The whitepaper bibliography. Every claim made in marketing copy must
trace back to one of these (or an equivalent peer-reviewed source).

| Domain | Foundational citation |
|---|---|
| Affect attunement | Stern, D. (1985). *The Interpersonal World of the Infant*. Basic Books. |
| Interpersonal neurobiology | Siegel, D. (2010). *The Mindful Therapist*. Norton. |
| Polyvagal theory | Porges, S. (2011). *The Polyvagal Theory*. Norton. |
| Affect regulation | Schore, A. (2003). *Affect Regulation and the Repair of the Self*. Norton. |
| Cultural attunement | Pedersen, P., Crethar, H., & Carlson, J. (2008). *Inclusive Cultural Empathy*. APA. |
| Cross-language information rate | Pellegrino, F., Coupé, C., & Marsico, E. (2011). *A cross-language perspective on speech information rate*. *Language* 87(3). |
| Facial asymmetry → genuineness | Delor, B., D'Hondt, F., & Philippot, P. (2021). *The Influence of Facial Asymmetry on Genuineness Judgment*. *Frontiers in Psychology* 12, 727446. |
| Laryngeal asymmetry → voice quality | Pillutla, S., Long, J. L., & Chhetri, D. K. (2022). *Effects of Laryngeal Vibratory Asymmetry and Neuromuscular Compensation on Voice Quality*. *The Laryngoscope* 132(1): 130-134. |
| Demographic bias in face AI | Buolamwini, J. & Gebru, T. (2018). *Gender Shades*. PMLR 81. |
| Whisper speech recognition | Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356. |
| MediaPipe landmarks | MediaPipe team (2023). *MediaPipe Tasks: On-device ML solutions*. Google Developers. |

---

## 7. Privacy posture (the moat)

| Component | Where it runs | Network at runtime |
|---|---|---|
| Video + audio capture | local | none |
| Audio extraction (ffmpeg) | local | none |
| Face landmarks (MediaPipe Tasks) | local TFLite | one-time model download |
| Hand landmarks (MediaPipe Tasks) | local TFLite | one-time model download |
| Speech transcript (faster-whisper) | local CTranslate2 int8 | one-time model download |
| Syllable-rate / VAD (librosa) | local | none |
| Syntonia Model math + change detection | local NumPy | none |
| Dashboard (Streamlit) | localhost only | telemetry hard-disabled |
| Caches and overlays | local disk only | none |

**Differentiator language for marketing**: "No cloud. No subscription. No
data leaves your device." Backed by `PRIVACY.md`, `scripts/check_isolation.py`,
`.streamlit/config.toml`, the ephemeral mode of `precompute_syntonia.py`.

---

## 8. Brand colour, typography, visual identity

**Status: deferred — separate creative engagement.**

Holding placeholder language until a visual identity is commissioned:

- Tonal register: clinical-warm. Sophisticated, not sterile. Scientific, not cold.
- Avoid: medical-clinical iconography (no stethoscopes, no microscopes),
  hyper-corporate vector illustrations, fitness-app green/orange palettes,
  AI-tech purple gradients, brain-with-circuits imagery.
- Likely register: soft neutrals, a single warm accent, generous whitespace,
  serif or humanist sans-serif for headlines, monospace for technical readouts.

---

## 9. Open decisions still pending

These are intentionally open and will be resolved as inputs arrive.

| Decision | Blocking on | Notes |
|---|---|---|
| Final tagline (1 of 4 candidates above) | Founder gut + 5–10 user reactions | Pick after first beta footage exists |
| Visual identity / logo / colour | Hire a designer post-trademark filing | $1–3k expense for a working design system |
| USPTO 1B intent-to-use filing | Founder decision; $700 cost | Recommended within 3 months of public launch |
| Pricing model | Initial trial cohort feedback | One-time vs. subscription, with PPP-adjusted regional tiers |
| Mac mini purchase | Founder decision | Refurbished M2 16 GB recommended; ~$679 |

---

## 10. Cross-references

- **Whitepaper**: `marketing/cultural_baseline_whitepaper.md` — technical defence of per-subject-baseline approach
- **Demo specification**: `marketing/demo_specification.md` — protocol for Yoodli-vs-SyntoniaPro comparison
- **Landing page hero**: `marketing/landing_page_hero.md` — multilingual launch copy
- **Privacy documentation**: `PRIVACY.md` — data-flow inventory
- **Privacy preflight script**: `scripts/check_isolation.py` — verifies project sits outside cloud-sync roots
- **Model fetch**: `scripts/fetch_models.py` — one-time download of MediaPipe + Whisper
- **Pipeline driver**: `scripts/precompute_syntonia.py` *(renamed from `precompute_syntonia.py`)*
- **Real-time analytics**: `dashboard.py` — Streamlit dashboard
- **Synthetic illustration**: `scripts/cultural_baseline_demo.py` — generates the per-subject vs. population threshold plot

---

## Change log

| Date | Decision |
|---|---|
| 2026-05-18 | Initial brand architecture committed. Consumer brand: SyntoniaPro. Technical engine: Syntonia Model. Domain SyntoniaPro.com purchased. USPTO TESS Classes 9 + 42 cleared. |
| 2026-05-18 | Codebase rename executed. `dfi.py → syntonia_model.py`. Symbols renamed: `DFIReport → SyntoniaReport`, `DFIWindow → SyntoniaWindow`, `compute_dfi → compute_syntonia`, `DFI_DISCLAIMER → SYNTONIA_DISCLAIMER`. Output file naming: `*_dfi_cache.npz → *_syntonia_cache.npz` and analogous for `_report.png` / `_events.json` / `_overlay.mp4`. Dashboard page title and headings rebranded. Marketing artifacts (whitepaper, demo specification, landing-page hero) updated to consistently use SyntoniaPro (consumer) and Syntonia Model (technical engine). 47 / 47 unit tests passing. Dashboard headless boot test: HTTP 200 in 4 s. |
