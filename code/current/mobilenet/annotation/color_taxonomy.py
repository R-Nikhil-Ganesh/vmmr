"""
The 11-color taxonomy shared by the teacher evaluation, the calibrator and the color head.

    white black silver red blue green yellow brown champagne pink_purple other

It is the CompCars human-label list (gray inside silver, beige/gold inside champagne, orange inside yellow) with
purple widened to pink_purple and an explicit `other`. Every source dataset maps into it:

  Qwen (13 colors)  QWEN_TO_11       gray -> silver, orange -> yellow, beige/gold -> champagne, other -> pink_purple
  CompCars (names)  COMPCARS_TO_11   purple -> pink_purple
  vCorR (folders)   VCORR_TO_11      grey -> silver, orange -> yellow, tan/beige/gold -> champagne, pink/purple -> pink_purple

Qwen has no pink/purple option, so its `other` is the only place a pink or purple car can land and it is read as
pink_purple. `other` has no human ground truth anywhere (no CompCars or vCorR rows), so no map emits it: it is
meant to be assigned by rule (low calibrated confidence or low option mass) and reported separately.
"""

COLORS11 = ["white", "black", "silver", "red", "blue", "green", "yellow", "brown", "champagne", "pink_purple", "other"]

QWEN_TO_11 = {
    "white": "white", "black": "black", "silver": "silver", "gray": "silver",
    "red": "red", "blue": "blue", "green": "green",
    "yellow": "yellow", "orange": "yellow",
    "brown": "brown", "beige": "champagne", "gold": "champagne",
    "other": "pink_purple",
}

COMPCARS_TO_11 = {
    "white": "white", "black": "black", "silver": "silver", "red": "red", "blue": "blue", "green": "green",
    "yellow": "yellow", "brown": "brown", "champagne": "champagne", "purple": "pink_purple",
}

VCORR_TO_11 = {
    "white": "white", "black": "black", "silver": "silver", "grey": "silver",
    "red": "red", "blue": "blue", "green": "green",
    "yellow": "yellow", "orange": "yellow",
    "brown": "brown", "beige": "champagne", "tan": "champagne", "gold": "champagne",
    "pink": "pink_purple", "purple": "pink_purple",
}


# "other" means pink_purple in the Qwen 13 list but the catch-all in the 11, so there is deliberately no generic
# name -> 11 function: pick the map for the source you are converting.
