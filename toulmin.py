"""Détection simple des composantes argumentatives selon le schéma de Toulmin."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = BASE_DIR / "dictionnaires" / "argumToulmin.json"


def normaliser_texte(texte: str) -> str:
    """Homogénéise apostrophes et espaces."""
    if not texte:
        return ""
    t = texte.replace("’", "'").replace("`", "'")
    t = re.sub(r"\s+", " ", t, flags=re.M)
    return t.strip()


def segmenter_phrases(texte: str) -> List[str]:
    """Segmente grossièrement en phrases à partir de la ponctuation forte."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]


def _motif_entier(expression: str) -> re.Pattern:
    expr_norm = re.escape(expression.replace("’", "'"))
    return re.compile(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){expr_norm}(?![A-Za-zÀ-ÖØ-öø-ÿ])", flags=re.I)


def charger_lexiques_toulmin(json_path: Any = None) -> Dict[str, List[str]]:
    """Charge le fichier JSON contenant les marqueurs (par défaut dans dictionnaires/)."""
    path = Path(json_path) if json_path is not None else DEFAULT_JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: [normaliser_texte(v) for v in vals] for k, vals in data.items()}


def compiler_motifs_toulmin(lexiques: Dict[str, Iterable[str]]) -> Dict[str, List[Tuple[str, re.Pattern]]]:
    motifs: Dict[str, List[Tuple[str, re.Pattern]]] = {}
    for categorie, expressions in lexiques.items():
        exprs_tries = sorted(set(expressions), key=lambda s: len(s), reverse=True)
        motifs[categorie] = [(expr, _motif_entier(expr)) for expr in exprs_tries if expr]
    return motifs


def detecter_toulmin(texte: str, lexiques: Dict[str, Iterable[str]] | None = None) -> List[Dict[str, str]]:
    """Retourne les occurrences par composante de Toulmin dans le texte."""
    if not texte:
        return []
    lexiques = lexiques or charger_lexiques_toulmin()
    motifs = compiler_motifs_toulmin(lexiques)
    phrases = segmenter_phrases(normaliser_texte(texte))

    resultats: List[Dict[str, str]] = []
    for idx, phrase in enumerate(phrases, start=1):
        for categorie, exprs in motifs.items():
            for expr, pattern in exprs:
                if pattern.search(phrase):
                    resultats.append(
                        {
                            "id_phrase": str(idx),
                            "categorie": categorie.replace("MARQUEURS_", "").title(),
                            "marqueur": expr,
                            "phrase": phrase,
                        }
                    )
    return resultats


__all__ = [
    "charger_lexiques_toulmin",
    "compiler_motifs_toulmin",
    "detecter_toulmin",
    "normaliser_texte",
    "segmenter_phrases",
]
