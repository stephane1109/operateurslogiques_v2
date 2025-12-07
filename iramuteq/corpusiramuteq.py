"""Outils de lecture et de segmentation de corpus IRaMuTeQ."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

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


def frequences_marqueurs_par_modalite(
    detections: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Calcule la fréquence des marqueurs logiques pour une modalité donnée."""

    def _compter(df: pd.DataFrame, colonne: str, type_label: str) -> pd.DataFrame:
        if df is None or df.empty or colonne not in df:
            return pd.DataFrame()
        freq = df[colonne].value_counts().reset_index()
        freq.columns = ["categorie", "frequence"]
        freq["type"] = type_label
        return freq

    tableaux = [
        _compter(detections.get("df_conn"), "code", "Connecteur logique"),
        _compter(detections.get("df_marq"), "categorie", "Marqueur normatif"),
        _compter(detections.get("df_memoires"), "categorie", "Mémoire"),
        _compter(detections.get("df_consq_lex"), "categorie", "Conséquence"),
        _compter(detections.get("df_causes_lex"), "categorie", "Cause"),
        _compter(detections.get("df_tensions"), "tension", "Tension sémantique"),
    ]

    tableaux = [t for t in tableaux if not t.empty]
    if not tableaux:
        return pd.DataFrame(columns=["type", "categorie", "frequence"])

    df_freq = pd.concat(tableaux, ignore_index=True)
    df_freq = df_freq[["type", "categorie", "frequence"]]
    return df_freq.sort_values(by=["type", "categorie"]).reset_index(drop=True)
