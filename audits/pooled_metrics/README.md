# Pooled metrics

`phystok.evaluation.metrics.pooled` computes RMSE=sqrt(sum(SSE)/sum(n)) and MAE=sum(SAE)/sum(n). `aggregate` requires all ten folds, three seeds and five populations, then reports mean and sample SD across complete seed metrics.

Mean city RMSE and fold SD are different estimands and cannot replace the main pooled results. Persistence has no training seed SD. Both `tests/test_training_and_audits.py` and the released aggregate verifier test these distinctions with unequal-support counterexamples.
