# DFI vs. Cloud-Coach Comparison Demo Protocol

A repeatable, defensible side-by-side comparison of the DFI pipeline against
a cloud incumbent (Yoodli is the primary target; the same protocol applies
to Poised / Orai). The output is a single page that demonstrates DFI's
per-subject-baseline + multilingual posture catches signal that
population-norm tools misinterpret.

This is a **marketing-asset production protocol**, not a scientific study.
It's designed to produce defensible side-by-side evidence — not statistical
proof. A formal validation study (Tier 2 pre-clinical) follows a different
protocol with IRB approval.

## Why this demo matters

| Claim it lets you make | Evidence the demo produces |
|---|---|
| "Western-tuned coach apps misread non-Western candidates" | Yoodli's actual report flags X behaviour as a problem; that behaviour is the subject's normal baseline |
| "Per-subject baseline eliminates that bias" | DFI's report for the same clip does not flag the same behaviour |
| "Multilingual ASR works on-device" | Whisper-small transcribes the subject's L1 (Hindi / Japanese / etc.) cleanly inside the DFI pipeline; competitor either can't handle the language or sends audio cross-border |
| "Privacy posture survives a global candidate's threat model" | Subject's clip never leaves their laptop in the DFI run; observable in network logs |

## Subjects to recruit

Recruit **3 subjects** across the global persona spectrum. Each subject
records the same scenario twice — once in English (the comparison language),
once in their L1 (to demonstrate the multilingual feature).

| # | Persona | Native language | Test scenario language(s) | Why this subject |
|---|---|---|---|---|
| 1 | South Asian L2 English speaker | Hindi / Tamil / Bengali | English + Hindi | Tests pace-norm bias (most documented in cloud tools) |
| 2 | East Asian baseline-expressiveness subject | Japanese / Mandarin | English + Japanese/Mandarin | Tests facial-baseline bias (well-documented) |
| 3 | Latin / Mediterranean gesture-rich subject | Italian / Spanish / Portuguese | English + L1 | Tests fidget-misclassification (gesture vocabulary) |

For each subject:
- Native speaker of their L1, conversational-or-better in English
- Voluntary participant, signs the demo-release form (template below)
- Compensated $50–100 per session
- Recordings used only for this single side-by-side comparison
- Subject sees both outputs before any public use; veto rights

**Anti-bias note**: Subjects should be coached *only* to perform a normal
practice session. Do not coach them to behave in ways that would flatter
either tool — the demo is about how each tool interprets normal behaviour.

## Recording protocol

**Equipment**: subject's own laptop or one we provide. Built-in webcam +
microphone. No external lighting (real-world condition).

**Environment**: subject's own home/office. Not a studio.

**Recording length**: 90 seconds per take. Format: 1080p MP4 from any
standard recording app.

### The script — same prompts for every subject, in their chosen language

The subject reads the following on-camera, paced naturally:

```
[Practice intro, 15 s — becomes baseline]
"Hi, I'm <name>. I'm a <role> from <city>. Today I'm practising for an
upcoming senior-engineer interview at a multinational technology company.
The interview will be conducted in English, which is my <Nth> language
after <L1>, <L2>."

[Behavioural-question response, 60 s]
"Tell me about a time you disagreed with a teammate.
... [the subject answers in 60 seconds, naturally, no script]"

[Closing, 15 s]
"Thanks for the practice. I'm going to review where I broke composure
and try again."
```

### Two takes per subject

**Take A** — English (the interview language). Subject pretends the
interviewer is in front of them.

**Take B** — Subject's L1. Same script, translated by the subject.

This produces 6 total clips across 3 subjects.

## Comparison procedure

For each clip:

1. **Upload to Yoodli** (or whichever cloud tool you're comparing against).
   Capture screenshots of:
   - The "overall score" or summary panel
   - The pace / filler-word / energy metrics
   - Any flagged moments with timestamps
   - Total session duration billed
2. **Run through DFI** locally:
   ```
   python scripts/precompute_dfi.py videos/<subject>_<lang>.mp4 \
       --baseline-seconds 12
   ```
   Open the dashboard. Capture screenshots of:
   - The state-panel readout at peak DFI moment
   - The breach event card (with verbatim transcript)
   - The AU cluster heatmap during the peak
3. **Network audit during the runs**:
   - Yoodli: run `tcpdump` or Wireshark during upload. Capture how much
     data crossed the border, to which IPs (their AWS region), and how long
     it was retained per their ToS.
   - DFI: run the same capture. Show zero outbound traffic (model already
     fetched).
4. **Generate the comparison sheet** (template in `marketing/comparison_sheet_template.md`).

## Fair-comparison principles

These are critical for credibility — anyone reading the comparison should
trust we didn't rig it.

| Rule | Reason |
|---|---|
| Same exact MP4 file fed to both tools | Eliminates "different recording quality" objections |
| Default settings on both tools | We don't tune DFI parameters to flatter ourselves |
| Both reports shown in full, unedited | No cherry-picking; show DFI's misses too |
| Subject confirms which report matches their experience | Subject is the ground truth |
| Yoodli's privacy claim shown verbatim from their ToS | We don't paraphrase their position into a strawman |
| All raw clips + reports archived | Anyone disputing the demo can re-run |

## Output: a single 1-page comparison sheet (per subject + language)

The artifact for each (subject × language) cell, suitable for marketing:

```
SUBJECT: <name>, <country>, <native language>
LANGUAGE OF RECORDING: <English | L1>

What the subject did:
- 90 s practice for "Tell me about a time you disagreed with a teammate"
- Normal pacing, gestures, facial expressions for them

What CLOUD-COACH-X concluded:
- Overall score: <X/100>
- Flagged: [list of flagged moments with timestamps + tool's interpretation]
- Recommendation: [tool's stated advice]
- Audio uploaded to: <ToS region>
- Audio retained: <ToS retention policy>

What DFI concluded:
- Max DFI: <X.X> (threshold 3.0)
- Breach events: <N>
- Per-frame channels: V=<peak> F=<peak> M=<peak>
- Verbatim line at peak: "<the subject's sentence>"
- Data sent off-device: 0 bytes

Subject's own assessment:
- Which report matched their experience? <CLOUD-X | DFI | both | neither>
- Any moment they wish either tool had caught? <subject's words>
- Privacy preference: <subject's words>
```

## Marketing-asset extraction

From a single 6-clip session you get:

1. **3 individual comparison sheets** (one per subject) — embeddable in
   blog posts, sales decks, landing-page case studies
2. **1 aggregate post**: "We tested Yoodli on three non-Western candidates —
   here's what it missed" (the public-facing blog format)
3. **3 quotable subject testimonials** — usable on the landing page
4. **3 short-form clips** of the DFI dashboard catching a moment the
   cloud tool missed (15-second loops, suitable for LinkedIn, Twitter)
5. **A network-traffic screenshot** showing 0 outbound bytes from the DFI
   run — the strongest single privacy proof

## Release form (one page, plain language)

Subjects sign before recording. Template:

> I, <name>, voluntarily participated in a comparison demo of behavioural
> analysis tools on <date>. I understand that:
> - My recording will be used in marketing materials for the DFI product,
>   in the configuration we agreed to.
> - I will see both reports before any public use, and may withdraw consent
>   at any time before publication.
> - The recording will be stored locally on <controller's> laptop and
>   deleted from cloud tools after the comparison.
> - I am compensated $<amount> for this participation.

## What this demo does NOT prove

Important to be honest in the marketing copy:

- It is not a controlled statistical study. N=3.
- It does not prove Yoodli is "wrong" in the general case — only that for
  these specific subjects, the population norms misinterpret their normal.
- It does not prove DFI's interpretations are clinically correct — only
  that they don't carry the same cultural-baseline bias.
- It does not prove DFI is more accurate at detecting actual deception or
  clinical conditions. That's a different study.

Frame the demo as a **bias illustration**, not a head-to-head accuracy claim.

## Cost + time estimate

| Phase | Time | Cost |
|---|---|---|
| Subject recruitment | 1–2 weeks | $0 (network), $300 (subject compensation) |
| Recording sessions | 1 day | $0 |
| Cloud tool licensing | $0 — use free tiers for the comparison | $0 |
| Analysis + comparison sheet production | 2 days | $0 |
| Blog / asset writing | 1 week | $0 |
| Total | ~3 weeks elapsed | **~$300 cash** |

A single round produces enough material to anchor 2–3 months of marketing.
