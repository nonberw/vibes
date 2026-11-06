# Submission package and checklist

Nothing has been submitted anywhere. Submitting requires the author's own accounts (OpenReview,
arXiv), which this repository does not touch.

## Venues (checked 2026-11-05/06)

| Venue | Deadline | Status for this paper |
|---|---|---|
| **ICML 2027** (main track) | Not yet announced on icml.cc (conference Aug 2-6, 2027). Aggregators project late January 2027 (unofficial). | **Recommended target.** The paper is formatted with the ICML 2026 style as a placeholder; switch to the 2027 style file when the CFP appears. |
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

1. When the ICML 2027 CFP is out, download its style file, replace `paper/icml2026.sty`/`.bst`,
   change `\usepackage[preprint]{icml2026}` to the blind-submission form of the 2027 package and
   re-run `tectonic main.tex`. Check the page limit (ICML 2026 allowed 8 pages of main text plus
   unlimited references and appendices).
2. Anonymize: remove the GitHub URL and the author line from `main.tex` for the blind version
   (the ICML style strips authors only in the `[accepted]` form; the URL is in the abstract and
   affiliation and must be removed by hand). Publish the anonymized code copy if you want
   reviewers to see it (e.g. an anonymous GitHub mirror).
3. Re-read `results/v2/hypotheses.md` and make sure every claim in the abstract matches the Holm
   outcomes. Every `% FINALIZE` comment in `paper/results.tex` must be resolved.
4. arXiv (optional, after or alongside submission, subject to the venue's policy): upload the
   `paper/` sources (tex, bib, pdf figures, style files). Category: cs.LG, cross-list cs.CV.
5. OpenReview: create the submission, upload the PDF, fill in the reciprocal reviewing fields,
   and the ethics/broader-impact statement (ICML requires an impact statement section; add it
   before submission).
6. Keep the public repository frozen at the submitted commit; tag it `icml2027-submission`.
