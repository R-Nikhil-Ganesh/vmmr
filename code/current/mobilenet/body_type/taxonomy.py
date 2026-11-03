"""
Vehicle Body Type (Car Type) Taxonomy and Normalization Utilities.

Standard 8-Class Canonical Taxonomy:
  0: sedan
  1: suv
  2: hatchback
  3: coupe
  4: convertible
  5: wagon
  6: pickup
  7: van
"""

from typing import Optional, Dict

BODY_TYPES = [
    "sedan",
    "suv",
    "hatchback",
    "coupe",
    "convertible",
    "wagon",
    "pickup",
    "van",
]

BODY_TYPE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(BODY_TYPES)}
IDX_TO_BODY_TYPE: Dict[int, str] = {i: name for i, name in enumerate(BODY_TYPES)}
NUM_BODY_TYPES: int = len(BODY_TYPES)


# CompCars type mapping (attributes.txt integer -> canonical body type)
COMPCARS_TYPE_MAP: Dict[int, str] = {
    1: "van",          # MPV
    2: "suv",          # SUV
    3: "sedan",        # sedan
    4: "hatchback",    # hatchback
    5: "van",          # minibus
    6: "coupe",        # fastback
    7: "wagon",        # estate
    8: "pickup",       # pickup
    9: "convertible",  # hardtop convertible
    10: "coupe",       # sports
    11: "suv",         # crossover
    12: "convertible", # convertible
}


# BoxCars116k type mapping
BOXCARS_TYPE_MAP: Dict[str, str] = {
    "sedan": "sedan",
    "suv": "suv",
    "hatchback": "hatchback",
    "combi": "wagon",
    "mpv": "van",
    "van": "van",
}


def normalize_body_type(text: str) -> Optional[str]:
    """Normalizes raw body style strings into one of the 8 canonical categories."""
    if not text:
        return None
    s = text.lower().strip()

    # Exact matches
    if s in BODY_TYPE_TO_IDX:
        return s

    # Aliases
    if s in ("saloon", "notchback"):
        return "sedan"
    if s in ("crossover", "compact suv", "4x4", "off-roader", "cuv"):
        return "suv"
    if s in ("hatch", "liftback"):
        return "hatchback"
    if s in ("fastback", "sports car", "sports", "supercar", "gt"):
        return "coupe"
    if s in ("cabriolet", "roadster", "spider", "spyder", "targa", "drophead"):
        return "convertible"
    if s in ("estate", "touring", "combi", "avant", "variant", "allroad", "station wagon", "shooting brake"):
        return "wagon"
    if s in ("truck", "utility truck", "cab", "extended cab", "crew cab", "regular cab", "quad cab", "supercab", "ute"):
        return "pickup"
    if s in ("minivan", "mpv", "minibus", "cargo van", "passenger van", "camper", "people mover"):
        return "van"

    # Substring search in order of specificity
    if "convertible" in s or "cabriolet" in s or "roadster" in s or "spider" in s:
        return "convertible"
    if "pickup" in s or "truck" in s or "cab" in s:
        return "pickup"
    if "wagon" in s or "estate" in s or "combi" in s or "avant" in s or "touring" in s or "variant" in s:
        return "wagon"
    if "minivan" in s or "mpv" in s or "van" in s:
        return "van"
    if "suv" in s or "crossover" in s:
        return "suv"
    if "hatchback" in s or "hatch" in s or "liftback" in s:
        return "hatchback"
    if "coupe" in s:
        return "coupe"
    if "sedan" in s or "saloon" in s:
        return "sedan"

    return None
