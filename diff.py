from diff_match_patch import diff_match_patch
from typing import Tuple

def find_diff(text1:str, text2:str, out_format:str='tuple') -> Tuple:
    dmp = diff_match_patch()
    dmp.Diff_EditCost = 4

    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)
    dmp.diff_cleanupEfficiency(diffs)

    if out_format == 'html' : 
        dmp.diff_prettyHtml(diffs)

    return diffs