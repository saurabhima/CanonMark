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
