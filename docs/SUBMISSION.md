# Submission package and checklist

Nothing has been submitted anywhere. Submitting requires the author's own accounts (OpenReview,
arXiv), which this repository does not touch.

## Venues (checked 2026-11-05/06)

| Venue | Deadline | Status for this paper |
|---|---|---|
| **ICML 2027** (main track, Rio de Janeiro, Jul 31 - Aug 6, 2027) | **Official CFP (icml.cc, read 2026-11-20):** submission site opens Jan 8, 2027; abstract deadline **Jan 14, 2027 11:59 PM AoE**; full paper **Jan 22, 2027 11:59 PM AoE**; no extensions. First-stage rejections Feb 23, 2027; decisions May 3, 2027. 8 pages main text + unlimited references, impact statement and appendices; double-blind; required Impact Statement and AI Disclosure Statement; a reciprocal reviewer must be named per submission; the CFP warns of submission-process updates to be announced late November 2026. | **Recommended target.** The paper already uses the official `icml2027` style (`paper/icml2027.sty`, downloaded from media.icml.cc) and has both required statements. |
| **ICLR 2027 workshops** (San Francisco, Apr 29-30, 2027) | Accepted workshops announced Nov 29, 2026; suggested paper deadline Feb 1, 2027; notification Feb 26, 2027. | **Companion / fallback.** Pick a workshop on world models, physical reasoning or interpretability once the list is out. Most ICLR workshops are non-archival; check the chosen workshop's and ICML's dual-submission rules before submitting to both. |
| CVPR 2027 | Nov 16, 2026 (registration Nov 10) | Too soon. |
| L4DC 2027 | Nov 13, 2026 (firm) | Too soon. |
| NeurIPS 2027 | Expected May 2027 | Fallback if ICML reviews are negative. |

## What is in the package

- `paper/main.pdf`: the paper, compiled with Tectonic from `paper/main.tex`.
- `paper/`: all sources, the ICML style, `refs.bib`, figures, and generated `numbers.tex`,
  `tables_*.tex` (from `scripts/paper_tables.py`, so every number comes from `results/`).
- `PREREGISTRATION.md`, `PREREGISTRATION_v2.md`: pre-registrations with dated deviation logs;
  `git log` has the timestamps.
- `results/v2/summary.json`, `results/v2/hypotheses.md`: aggregated statistics and test outcomes.
- Reproduction: `README.md` (commands), `requirements.txt`, `tests/`.

## Checklist for the author

1. For the blind submission change `\usepackage[preprint]{icml2027}` to `\usepackage{icml2027}`
   in `paper/main.tex` and re-run `tectonic main.tex`. Check that the main text (before References)
   is at most 8 pages. Re-read the CFP before submitting: ICML announced submission-process updates
   for late November 2026 (format unchanged).
2. Anonymize: remove the GitHub URL and the author line from `main.tex` for the blind version
   (the ICML style strips authors only in the `[accepted]` form; the URL is in the abstract and
   affiliation and must be removed by hand). Publish the anonymized code copy if you want
   reviewers to see it (e.g. an anonymous GitHub mirror).
3. Re-read `results/v2/hypotheses.md` and make sure every claim in the abstract matches the Holm
   outcomes. Every `% FINALIZE` comment in `paper/results.tex` must be resolved.
4. arXiv (optional, after or alongside submission, subject to the venue's policy): upload the
   `paper/` sources (tex, bib, pdf figures, style files). Category: cs.LG, cross-list cs.CV.
5. OpenReview (opens Jan 8, 2027; create the account early): register the abstract by Jan 14,
   upload the PDF by Jan 22, name the reciprocal reviewer (yourself), rank your submissions if you
   have several, and fill in the AI disclosure fields consistently with `paper/disclosure.tex`
   (the paper was written with an AI coding agent; ICML requires this to be disclosed).
6. Keep the public repository frozen at the submitted commit; tag it `icml2027-submission`.
