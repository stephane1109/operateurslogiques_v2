import json
import re
from pathlib import Path
from typing import List, Dict

# --- utils ---
def segment_sentences(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?<=[\.!\?:;])\s+", text) if p.strip()]

def compile_list(patterns: List[str], use_regex: bool):
    if use_regex:
        return [re.compile(p, flags=re.I) for p in patterns]
    else:
        # sécurise en mots entiers si ce ne sont pas des regex
        return [re.compile(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(p)}(?![A-Za-zÀ-ÖØ-öø-ÿ])", flags=re.I) for p in patterns]

def any_match(compiled_patterns, text: str) -> bool:
    return any(p.search(text) for p in compiled_patterns)

# --- charge le JSON ---
BASE_DIR = Path(__file__).resolve().parent
with (BASE_DIR / "dictionnaires" / "tension_semantique.json").open(
    "r", encoding="utf-8"
) as f:
    TENS = json.load(f)

USE_REGEX = bool(TENS.get("options", {}).get("patterns_are_regex", True))
WINDOW_TYPE = TENS.get("options", {}).get("window_type", "sentence")
WIN_TOK = int(TENS.get("options", {}).get("sliding_window_tokens", 50))

# compile pôles
POLE_PATTERNS: Dict[str, List[re.Pattern]] = {
    pole: compile_list(patterns, USE_REGEX) for pole, patterns in TENS["poles"].items()
}
# compile connecteurs contraste
CONTRAST = compile_list(TENS.get("contrast_connectors", []), True)

def detect_tensions(text: str) -> List[Dict]:
    results = []
    sents = segment_sentences(text.replace("’", "'"))
    for i, sent in enumerate(sents, start=1):
        s = sent  # on peut normaliser davantage si besoin
        for pair in TENS["pairs"]:
            A, B, label = pair["left"], pair["right"], pair["label"]
            hasA = any_match(POLE_PATTERNS[A], s)
            hasB = any_match(POLE_PATTERNS[B], s)
            if hasA and hasB:
                strong = any_match(CONTRAST, s)
                results.append({
                    "id_phrase": i,
                    "tension": label,
                    "pole_A": A,
                    "pole_B": B,
                    "intensite": "forte" if strong else "faible",
                    "phrase": sent
                })
    return results

# Exemple d’usage :
# occs = detect_tensions(votre_discours)
# -> liste de dicts; vous pouvez faire un DataFrame, afficher un tableau, etc.
