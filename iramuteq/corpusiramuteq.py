"""Outils de lecture et de segmentation de corpus IRaMuTeQ."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd


def nettoyer_nom_modalite(modalite_brute: str) -> str:
    """Nettoie une modalité en supprimant les marqueurs et espaces superflus."""
    if modalite_brute is None:
        return ""
    return modalite_brute.strip().strip("*_ ")


def segmenter_corpus_par_modalite(texte_corpus: str) -> pd.DataFrame:
    """Segmente un corpus IRaMuTeQ en DataFrame modalité / texte.

    Le format attendu repose sur des lignes de type "**** *modalite_" qui
    marquent le début d'un nouveau segment. Tout le texte qui suit jusqu'à la
    prochaine balise appartient à cette modalité.
    """

    if not texte_corpus:
        return pd.DataFrame(columns=["modalite", "texte"])

    motif_modalite = re.compile(r"^\*{4}\s*\*(.+)$", re.MULTILINE)
    segments: List[dict] = []

    positions = list(motif_modalite.finditer(texte_corpus))
    for idx, match in enumerate(positions):
        debut_contenu = match.end()
        fin_contenu = positions[idx + 1].start() if idx + 1 < len(positions) else len(texte_corpus)
        contenu = texte_corpus[debut_contenu:fin_contenu].strip()
        modalite = nettoyer_nom_modalite(match.group(1))
        if modalite:
            segments.append({"modalite": modalite, "texte": contenu})

    return pd.DataFrame(segments, columns=["modalite", "texte"])


def filtrer_modalites(df_modalites: pd.DataFrame, modalites: Iterable[str]) -> pd.DataFrame:
    """Retourne uniquement les lignes correspondant aux modalités choisies."""
    if df_modalites is None or df_modalites.empty:
        return pd.DataFrame(columns=["modalite", "texte"])
    modalites_set = {m for m in modalites if m}
    if not modalites_set:
        return pd.DataFrame(columns=["modalite", "texte"])
    return df_modalites[df_modalites["modalite"].isin(modalites_set)].copy()


def fusionner_textes_modalites(df_modalites: pd.DataFrame) -> str:
    """Concatène les textes des modalités en les séparant par deux retours."""
    if df_modalites is None or df_modalites.empty:
        return ""
    textes = [str(val).strip() for val in df_modalites["texte"] if str(val).strip()]
    return "\n\n".join(textes)
