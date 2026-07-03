# GenoSmart neutral partial-test-leak importance run

This run treats VASN as an ordinary feature. There is no VASN-specific boost, head, loss, or post-hoc rank injection.

Training used train + validation + part of the original test set, and evaluation was reported on the original full test. These metrics are exploratory and not independent validation.

- VASN fixed XGB token position: 568
- leaked test samples used in training: 62 / 69
- holdout test samples not used in training: 7 / 69
- best epoch selected by full-test ROC-AUC: 7
- full-test ROC-AUC OvR: 0.980972
- full-test accuracy: 0.811594
- full-test macro-F1: 0.819988
- full-test class AUCs: 0.978151, 0.974946, 0.989819
- holdout-test ROC-AUC OvR: 0.555556
- holdout-test accuracy: 0.428571
- holdout-test macro-F1: 0.333333
- checkpoint: `genosmart_neutral_partial_test_best.pt`
- history: `history.csv`
- csv table: `xgb_top2000_neutral_transformer_importance.csv`
- excel table: `xgb_top2000_neutral_transformer_importance.xlsx`
- VASN summary: `vasn_rank_summary.csv`

## Feature-importance methods

- embedding_grad_abs: absolute gradient saliency on each gene embedding token.
- embedding_grad_x_input: gradient x input on gene embedding tokens.
- weight_grad_abs: absolute gradient saliency on each expression weight scalar.
- weight_grad_x_input: gradient x input on expression weight scalar.
- integrated_grad_x_input: integrated gradients from zero baseline on embedding and weight inputs.
- mean_expression_weight: average input expression weight across full-test samples.

## VASN rank

- consensus rank: 146 / 2000
- embedding_grad_abs rank: 1309 / 2000
- embedding_grad_x_input rank: 188 / 2000
- weight_grad_abs rank: 1211 / 2000
- weight_grad_x_input rank: 144 / 2000
- integrated_grad_x_input rank: 283 / 2000
- mean_expression_weight rank: 189 / 2000
