

def result2dict(results):
    score_dict = {}
    known_dict = {}
    scores = 0
    miss = 0
    tot = 0
    for name, (score, missing_words, total_words) in results.items():
        scores += score
        miss += missing_words
        tot += total_words
        score_dict[name] = score
        known_dict[name] = (total_words - missing_words) / total_words
    score_dict["Average"] = scores / len(results)
    known_dict["Average"] = (tot - miss) / tot
    return score_dict, known_dict
