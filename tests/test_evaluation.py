from canonmark.evaluation import mcnemar


def test_mcnemar_counts_and_exact_p():
    base = {f"t{i}": True for i in range(20)}
    wm = dict(base)
    for i in range(6):          # watermark breaks 6 problems, fixes none
        wm[f"t{i}"] = False
    r = mcnemar(base, wm)
    assert (r["lost"], r["gained"], r["n_pairs"]) == (6, 0, 20)
    assert abs(r["delta_pass@1"] + 0.3) < 1e-9
    assert abs(r["p_value"] - 2 / 2 ** 6) < 1e-9        # two-sided exact: 2 * 0.5^6


def test_mcnemar_symmetric_and_no_discordance():
    a = {"x": True, "y": False, "z": True, "w": False}
    b = {"x": False, "y": True, "z": True, "w": False}
    assert mcnemar(a, b)["p_value"] == 1.0
    assert mcnemar(a, a)["p_value"] == 1.0 and mcnemar(a, a)["delta_pass@1"] == 0


def test_mcnemar_pairs_only_common_tasks():
    r = mcnemar({"a": True, "b": True}, {"b": False, "c": True})
    assert r["n_pairs"] == 1 and r["lost"] == 1


def test_auroc_and_bootstrap_ci():
    import random
    from canonmark.evaluation import auroc_ci, det_metrics, key_averaged_auroc_ci
    rng = random.Random(0)
    pos = [rng.gauss(0.8, 1) for _ in range(164)]
    neg = [rng.gauss(0.0, 1) for _ in range(164)]
    m = det_metrics(pos, neg, n_boot=500)
    from sklearn.metrics import roc_auc_score
    assert abs(m["auroc"] - roc_auc_score([1] * 164 + [0] * 164, pos + neg)) < 1e-4   # ties handled the same
    lo, hi = m["auroc_ci95"]
    assert lo < m["auroc"] < hi and 0.05 < hi - lo < 0.2          # ~±0.04-0.06 at n=164
    small = auroc_ci(pos[:30], neg[:30], 500)
    assert small[1] - small[0] > hi - lo                           # fewer problems -> wider interval
    run = {"pos": {f"t{i}": z for i, z in enumerate(pos)}, "neg_model": {f"t{i}": z for i, z in enumerate(neg)}}
    pt, ci = key_averaged_auroc_ci([run, run], n_boot=300)
    assert abs(pt - m["auroc"]) < 1e-4 and ci[0] < pt < ci[1]
