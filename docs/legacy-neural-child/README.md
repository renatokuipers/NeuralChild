# README.md

Top-level prose documentation for the legacy `neural-child` repository. It describes an intended developmental-AI system (stage-based psychological growth, emotional regulation, attachment, theory of mind) entirely in narrative and bulleted taxonomy form. It contains no executable content, no commands, no module or repo paths (the only file named anywhere is `LICENSE`, README.md:534), and no API description — it is an aspirational design document, not a usable orientation guide.

## Public surface

No code. The "surface" is the H2 section set; every one is descriptive only.

| Name | Kind | Contract (one line) |
|---|---|---|
| Table of Contents | section, README.md:3 | 15 anchor links; 2 point at headings that do not exist in the file |
| Introduction | section, README.md:21 | States the premise: network starts "newborn" and develops through stages |
| Theoretical Foundations | section, README.md:40 | Names Piaget, Bowlby/Ainsworth, emotional development, plasticity, memory, social brain |
| System Architecture | section, README.md:84 | Names 4 subsystems: sensory, emotional, memory, psychological components |
| Developmental Stages | section, README.md:130 | Names 3 stages with month ranges, then switches to capability taxonomy |
| Psychological Components | section, README.md:178 | Emotional regulation, defense mechanisms, theory of mind taxonomies |
| Memory and Learning | section, README.md:228 | Short-term / working / long-term memory; supervised, unsupervised, emotional learning |
| Model Performance | section, README.md:274 | Claims a trained model exists and lists evaluation dimensions; no metric values |
| Applications | section, README.md:320 | Research, education, therapeutic use cases (current + future) |
| Technical Implementation | section, README.md:366 | Only concrete section: hardware/software minimums |
| Future Research Directions | section, README.md:400 | Planned capability/technical/feature work |
| Ethics and Considerations | section, README.md:446 | Development and application ethics, technical and psychological safety |
| Getting Started | section, README.md:480 | Names setup topics; supplies no steps |
| Contributing | section, README.md:514 | Names code standards and process topics; supplies no rules |
| License | section, README.md:532 | MIT, defers to a LICENSE file (unverifiable from this file alone) |
| Citation | section, README.md:536 | BibTeX block, `neural_child_development`, year 2025 |
| Acknowledgments | section, README.md:550 | Lists 7 contributing fields |

## Key behaviour

- The only hard, machine-relevant facts in the entire file are the requirements at README.md:373-382: CUDA-capable GPU, minimum 16 GB RAM, SSD, multi-core CPU; Python 3.8+, PyTorch 1.8+, CUDA 11.0+, plus unnamed "Additional dependencies".
- Developmental stages are the only place with numeric ranges: Newborn 0-3 months (README.md:136), Early Infancy 3-6 months (README.md:142), Late Infancy 6-12 months (README.md:148). No stage beyond Late Infancy is enumerated; the remainder is elided by a one-line bracketed editorial note (README.md:154) before the section moves to Stage-Specific Capabilities (README.md:156).
- Claimed component graph (README.md:88-128) — assertions only, no module or class names given:

```
 Sensory Processing ──┐
 Emotional Network ───┼── bidirectional flow / state sync ──> integrated state
 Memory Systems ──────┤        (README.md:118-122)
 Psychological Comp. ─┘
        ^
        └── developmental plasticity: stage-appropriate learning rates,
            critical-period modulation (README.md:124-128)
```

- Defense mechanisms are enumerated as two tiers (README.md:198-210): primary = repression, denial, projection, regression; mature = sublimation, humor, anticipation, altruism.
- Attachment patterns enumerated at README.md:53-56: secure, anxious, avoidant, disorganized.
- No install command, no entry point, no repo layout, no configuration key, no environment variable, and no example invocation appears anywhere in the file.

## Imports

None. The file is Markdown prose with a single fenced BibTeX block (README.md:540-548). No third-party or project-internal imports exist.

## Defects and gaps

- Dead TOC anchors: README.md:11 links `#emotional-processing` and README.md:12 links `#training-methodology`, but no heading with either title exists in the file. "Training Methodology" has no content at all; emotional material lives under Psychological Components (README.md:180).
- TOC is incomplete in the other direction: License (README.md:532), Citation (README.md:536), and Acknowledgments (README.md:550) are absent from the 15-entry TOC.
- BibTeX placeholder residue: the author field at README.md:544 is wrapped in literal square brackets, and the URL at README.md:545 is malformed — an opening bracket is closed by a parenthesis followed by a bracket, so the emitted citation carries stray bracket and parenthesis characters inside the URL value.
- Section headers promise content they never deliver: README.md:386 and README.md:484 both say detailed setup instructions are provided, then list only topic names; README.md:300-318 claims performance "has been evaluated" but gives zero numbers, datasets, or protocols; README.md:278 asserts a trained model exists with no checkpoint, size, or artifact reference.
- README.md:382 lists "Additional dependencies" as a requirement without enumerating any, so the stated software prerequisites are not sufficient to reproduce an environment.
- Tense mismatch throughout: capabilities are stated in the present indicative ("The system implements", README.md:198) while nothing in the file distinguishes implemented behaviour from intended design. Nothing here can be used to confirm any subsystem exists.

## Notes

- Treat this file as a statement of intent only. Every architectural claim in it needs verification against actual modules before being carried into any current-repo documentation; none of it is verifiable from this file alone.
- The stated minimum stack (Python 3.8+, PyTorch 1.8+, CUDA 11.0+) is the one fact worth carrying forward, and even that is a floor, not necessarily the version actually used.
- Author is given as Renato Kuipers, repo path `renatokuipers/neural-child`, year 2025 (README.md:541-546).
