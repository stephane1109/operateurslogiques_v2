"""Fonctions utilitaires de normalisation et segmentation de texte."""

from __future__ import annotations

import re
from typing import List


def normaliser_espace(texte: str) -> str:
    """Homogénéise espaces et apostrophes afin de stabiliser les recherches."""
    if not texte:
        return ""
    t = texte.replace("’", "'").replace("`", "'")
    t = re.sub(r"\s+", " ", t, flags=re.M)
    return t.strip()


def segmenter_en_phrases(texte: str) -> List[str]:
    """Segmente approximativement en phrases sur ponctuation forte."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]
