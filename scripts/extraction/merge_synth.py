# Merge the 30 per-repetition result files sv_x18_r20.95_rep{0..29}.json (synth_solvent_one.py) into one JSON.
import json, glob, re, sys
runs = {int(re.search(r"rep(\d+)", f).group(1)): json.load(open(f)) for f in glob.glob(sys.argv[1] + "/sv_x18_r20.95_rep*.json")}
r0 = runs[0]
keep = ["data", "truth", "j_true", "r2", "reps", "beta", "sigma", "r2_eff", "boot", "max_size", "n_combos", "names", "n_rows", "corr_with_truth"]
out = {k: r0[k] for k in keep}
out["bolasso_freq"] = [runs[r]["bolasso_raw"][0] for r in range(30)]
out["esbma_pin"] = [runs[r]["esbma_raw"][0] for r in range(30)]
assert all(runs[r]["reps_here"] == [r] for r in range(30))
json.dump(out, open(sys.argv[2], "w"), indent=1); print(len(runs))
