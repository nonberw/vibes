# v2 pre-registered hypotheses: outcomes

One-sided hierarchical-bootstrap p-values; Holm over all confirmatory tests.

| Test | p (raw) | p (Holm) | Supported |
|---|---|---|---|
| H1_jepa | 0.0500 | 0.5000 | no |
| H1_rec | 0.6500 | 1.0000 | no |
| H1_state | 0.0000 | 0.0000 | yes |
| H2_jepa_hi | 0.6050 | 1.0000 | no |
| H2_jepa_lo | 1.0000 | 1.0000 | no |
| H2_rec_hi | 0.1500 | 1.0000 | no |
| H2_rec_lo | 1.0000 | 1.0000 | no |
| H2_state_hi | 0.2200 | 1.0000 | no |
| H2_state_lo | 1.0000 | 1.0000 | no |
| H3_jepa | 0.2700 | 1.0000 | no |
| H3_rec | 0.9650 | 1.0000 | no |
| H3_state | 0.0000 | 0.0000 | yes |
| H5_jepa | 0.0000 | 0.0000 | yes |
| H5_rec | 0.0000 | 0.0000 | yes |

## Locus of OOD failure (decision rule, Section 6)

| Config | gate | high side | low side | inv_h reproduces anti-physics |
|---|---|---|---|---|
| jepa_M_gru_train_100000 | True | mixed | mixed | False |
| rec_M_gru_train_100000 | True | mixed | mixed | False |
| rec_M_gru_wide_100000 | True | mixed | unreachable belief | True |
| state_M_gru_train_100000 | True | unreachable belief | mixed | False |
| state_M_gru_wide_100000 | True | mixed | mixed | True |
| state_M_lru_train_100000 | True | unreachable belief | mixed | False |
