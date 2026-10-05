from itertools import combinations, product


def score_with_metric(metric, sample, verbose=0, **kwargs):
    # print(metric.__name__)
    scores = dict()
    fails = dict()
    para_keys = [ k for k in sample if k.startswith("=") ]
    diff_keys = [ k for k in sample if not (k.startswith("=") or k=="id") ]

    # utility function for comparing pairs
    def compare_pairs(pair_combinations):
        key_pairs, sent1_list, sent2_list = [], [], []
        for k1, k2 in pair_combinations:
            key_pairs.append((k1, k2))
            sent1_list.append(sample[k1])
            sent2_list.append(sample[k2])
        score_list = metric(sent1_list, sent2_list, **kwargs)
        return dict(zip(key_pairs, score_list, strict=True))

    # score paraphrases first
    key_pairs_to_score = compare_pairs(combinations(para_keys, 2))
    scores.update(key_pairs_to_score)
    # score non-paraphrases
    key_pairs_to_score = compare_pairs(product(para_keys, diff_keys))
    scores.update(key_pairs_to_score)
    # check fails
    count = 0
    for (pk1, pk2), dk in product(combinations(para_keys, 2), diff_keys):
        count += 1
        pp_score = scores[(pk1, pk2)]
        p1_d_score = scores[(pk1, dk)]
        p2_d_score = scores[(pk2, dk)]
        if pp_score < p1_d_score or pp_score < p2_d_score:
            triple = {(pk1, pk2): pp_score, (pk1, dk): p1_d_score, (pk2, dk): p2_d_score}
            ranked_keys = sorted(triple, key=triple.get)
            fails[(pk1, pk2, dk)] = (pp_score, p1_d_score, p2_d_score)
            if verbose >= 2:
                print(f"❌ {metric.__name__} id={sample['id']}: {sample['=base']}")
                keys = " ".join([ f"{k}" for k in ranked_keys ])
                print(f"\t{keys}")
    fail_rate = len(fails)/count
    if verbose >= 1:
        print(f"‼️{metric.__name__}: {(len(fails)/count*100):.1f}% = {len(fails)}/{count}")
    return scores, fails, fail_rate


def check_metric(metric, data, verbose=0, **kwargs):
    print(metric.__name__)
    scores, fails = dict(), dict()
    fail_rates = []
    for sample in data:
        _scores, _fails, _fail_rate = score_with_metric(metric, sample, verbose=verbose, **kwargs)
        scores[sample["id"]] = _scores
        fails[sample["id"]] = _fails
        fail_rates.append(_fail_rate)
    mean_fail_rate = sum(fail_rates)/len(fail_rates)
    if True:
        print(f"‼️{metric.__name__}: mean fail rate = {mean_fail_rate*100:.1f}%")
    return scores, fails, mean_fail_rate