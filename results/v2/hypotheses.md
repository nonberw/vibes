# v2 pre-registered hypotheses: outcomes

One-sided hierarchical-bootstrap p-values; Holm over all confirmatory tests.

| Test | p (raw) | p (Holm) | Supported |
|---|---|---|---|
| H1_jepa | 0.1160 | 1.0000 | no |
| H1_rec | 0.6800 | 1.0000 | no |
| H1_state | 0.0000 | 0.0000 | yes |
| H2_jepa_hi | 0.6375 | 1.0000 | no |
| H2_jepa_lo | 1.0000 | 1.0000 | no |
| H2_rec_hi | 0.1250 | 1.0000 | no |
| H2_rec_lo | 1.0000 | 1.0000 | no |
| H2_state_hi | 0.2165 | 1.0000 | no |
| H2_state_lo | 1.0000 | 1.0000 | no |
| H3_jepa | 0.3160 | 1.0000 | no |
| H3_rec | 0.9780 | 1.0000 | no |
| H3_state | 0.0000 | 0.0000 | yes |
| H4_jepa | 0.7320 | 1.0000 | no |
| H4_rec | 0.2580 | 1.0000 | no |
| H4_state | 0.6485 | 1.0000 | no |
| H5_jepa | 0.0000 | 0.0000 | yes |
| H5_rec | 0.0000 | 0.0000 | yes |

## Locus of OOD failure (decision rule, Section 6)

| Config | gate | high side | low side | inv_h reproduces anti-physics |
|---|---|---|---|---|
| jepa_L_gru_train_100000 | True | unreachable belief | mixed | False |
| jepa_M_gru_train_1000 | True | mixed | mixed | False |
| jepa_M_gru_train_8000 | True | mixed | mixed | False |
| jepa_M_gru_train_100000 | True | mixed | mixed | False |
| jepa_M_gru_wide_100000 | True | mixed | mixed | False |
| jepa_M_lru_train_100000 | True | mixed | unreachable belief | False |
| jepa_S_gru_train_100000 | True | mixed | mixed | False |
| rec_L_gru_train_100000 | True | mixed | mixed | True |
| rec_M_gru_train_1000 | False | unreachable belief | unreachable belief | True |
| rec_M_gru_train_8000 | True | mixed | mixed | False |
| rec_M_gru_train_100000 | True | mixed | mixed | False |
| rec_M_gru_wide_100000 | True | mixed | unreachable belief | True |
| rec_M_lru_train_100000 | True | unreachable belief | unreachable belief | False |
| rec_S_gru_train_100000 | True | mixed | mixed | False |
| state_L_gru_train_100000 | True | mixed | mixed | True |
| state_M_gru_train_1000 | True | mixed | mixed | True |
| state_M_gru_train_8000 | True | mixed | mixed | False |
| state_M_gru_train_100000 | True | unreachable belief | mixed | False |
| state_M_gru_wide_100000 | True | mixed | mixed | True |
| state_M_lru_train_100000 | True | unreachable belief | mixed | False |
| state_S_gru_train_100000 | True | unreachable belief | mixed | False |
