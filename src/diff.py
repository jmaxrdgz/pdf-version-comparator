from diff_match_patch import diff_match_patch
from difflib import SequenceMatcher
from typing import List, Dict, AnyStr
from src.text import clean_sentence, segment_sentences

def substring_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()

def diff_texts(text1: str, text2: str, seq_threshold: float = 0.75) -> Dict[str, List[AnyStr]]:
    sentences1, sentences2 = segment_sentences(text1, text2)

    dmp = diff_match_patch()
    dmp.Match_Threshold = 0.4
    dmp.Match_Distance = 20

    unchanged = []
    used_in_sentences2 = set()

    for i, s1 in enumerate(sentences1):
        matches = [j for j, s2 in enumerate(sentences2) if s2 == s1 and j not in used_in_sentences2]
        if matches:
            unchanged.append(s1)
            used_in_sentences2.add(matches[0])

    remaining1 = [s for s in sentences1 if s not in unchanged]
    remaining2_indices = [j for j, s in enumerate(sentences2) if j not in used_in_sentences2]
    remaining2 = [sentences2[j] for j in remaining2_indices]

    matched_pairs = []
    used_r2 = set()

    for r1 in remaining1:
        r1_clean = clean_sentence(r1)
        best_match = None
        best_sim_score = 0.0
        best_index = None

        for idx, r2 in enumerate(remaining2):
            if idx in used_r2:
                continue
            r2_clean = clean_sentence(r2)
            match_pos = dmp.match_main(r2_clean, r1_clean, 0)
            if match_pos != -1:
                end_pos = match_pos + len(r1_clean)
                if end_pos > len(r2_clean):
                    end_pos = len(r2_clean)
                segment = r2_clean[match_pos:end_pos]

                sim = substring_similarity(r1_clean, segment)
                if sim > best_sim_score and sim >= seq_threshold:
                    best_sim_score = sim
                    best_match = r2
                    best_index = idx

        if best_match is not None:
            matched_pairs.append((r1, best_match))
            used_r2.add(best_index)

    matched_removed = {p[0] for p in matched_pairs}
    matched_added = {p[1] for p in matched_pairs}

    removed = [r1 for r1 in remaining1 if r1 not in matched_removed]
    added = [r2 for r2 in remaining2 if r2 not in matched_added]

    newly_modified = []
    remaining_added = added[:]
    remaining_removed = removed[:]

    for rem in removed:
        best_match = None
        best_score = seq_threshold
        for add in added:
            sim = substring_similarity(clean_sentence(rem), clean_sentence(add))
            if sim >= best_score:
                best_score = sim
                best_match = add
        if best_match is not None:
            newly_modified.append((rem, best_match))
            if best_match in remaining_added:
                remaining_added.remove(best_match)
            if rem in remaining_removed:
                remaining_removed.remove(rem)

    final_result = {
        "unchanged": unchanged,
        "modified": matched_pairs + newly_modified,
        "added": remaining_added,
        "removed": remaining_removed
    }

    return final_result

