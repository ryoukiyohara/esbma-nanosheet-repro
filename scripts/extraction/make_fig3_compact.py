# Reduce the arrays extracted from output/exp1/run2/model.pickle (extract_fig3.py) to what Fig 3 needs:
# marginal log-likelihood of all 65,536 models + coefficients of the 200 models shown in the weight diagrams.
import sys, numpy as np
z = np.load(sys.argv[1]); ll = z["loglik"]; order = np.argsort(ll)[::-1]
shown = np.r_[order[:100], order[59950:60050]]          # BFE ranks 1-100 and 59,951-60,050
ind = z["indicators"]; assert (ind == ((np.arange(len(ll))[:, None] >> np.arange(16)) & 1)).all()
np.savez_compressed(sys.argv[2], loglik=ll, shown_model_index=shown, shown_coef=z["coef"][shown],
                    feature_posteriors=z["feature_posteriors"])
print(len(ll), shown[:3])
