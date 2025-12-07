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


def extraire_variable_et_modalite(nom_balise: str) -> Dict[str, str]:
    """Retourne la variable et la modalité à partir d'une balise IRaMuTeQ."""

    nom_nettoye = nettoyer_nom_modalite(nom_balise)
    if not nom_nettoye:
        return {"variable": "", "modalite": ""}

    if "_" in nom_nettoye:
        variable, modalite = nom_nettoye.split("_", 1)
    else:
        variable, modalite = nom_nettoye, ""

    return {"variable": variable.strip(), "modalite": modalite.strip()}


def segmenter_corpus_par_modalite(texte_corpus: str) -> pd.DataFrame:
    """Segmente un corpus IRaMuTeQ en DataFrame variable / modalité / texte.

    Le format attendu repose sur des blocs introduits par une ligne "****" puis
    des lignes de modalités commençant par "*" (par exemple "*variable_modalite").
    L'ancien format "**** *variable_modalite" reste pris en charge. Chaque
    modalité regroupe le texte jusqu'à la prochaine modalité ou balise "****".
    """

    if not texte_corpus:
        return pd.DataFrame(columns=["variable", "modalite", "texte", "balise"])

    motif_balises = re.compile(r"^(?P<limite>\*{4}\s*$)|^(?P<balise>\*.+)$", re.MULTILINE)
    segments: List[dict] = []

    balises = list(motif_balises.finditer(texte_corpus))
    for idx, match in enumerate(balises):
        balise_modalite = match.group("balise")
        if not balise_modalite:
            continue  # Ligne "****" : on avance jusqu'à la prochaine balise utile

        debut_contenu = match.end()
        fin_contenu = balises[idx + 1].start() if idx + 1 < len(balises) else len(texte_corpus)
        contenu = texte_corpus[debut_contenu:fin_contenu].strip()

        infos_balise = extraire_variable_et_modalite(balise_modalite)
        if infos_balise.get("variable") or infos_balise.get("modalite"):
            segments.append(
                {
                    "variable": infos_balise.get("variable", ""),
                    "modalite": infos_balise.get("modalite", ""),
                    "texte": contenu,
                    "balise": balise_modalite.strip(),
                }
            )

    return pd.DataFrame(segments, columns=["variable", "modalite", "texte", "balise"])


def filtrer_modalites(
    df_modalites: pd.DataFrame, modalites: Iterable[str], variable: str | None = None
) -> pd.DataFrame:
    """Retourne uniquement les lignes correspondant aux modalités et variable choisies."""

    if df_modalites is None or df_modalites.empty:
        return pd.DataFrame(columns=["variable", "modalite", "texte", "balise"])

    df_filtre = df_modalites
    if variable:
        df_filtre = df_filtre[df_filtre["variable"] == variable]

    modalites_set = {m for m in modalites if m}
    if modalites_set:
        df_filtre = df_filtre[df_filtre["modalite"].isin(modalites_set)]

    if df_filtre.empty:
        return pd.DataFrame(columns=["variable", "modalite", "texte", "balise"])

    return df_filtre.copy()


def fusionner_textes_modalites(df_modalites: pd.DataFrame) -> str:
    """Concatène les textes des modalités en préservant les balises d'origine."""

    if df_modalites is None or df_modalites.empty:
        return ""

    segments_concat = []
    for _, ligne in df_modalites.iterrows():
        texte = str(ligne.get("texte", "")).strip()
        if not texte:
            continue

        balise = str(ligne.get("balise") or "").strip()
        if not balise:
            variable = str(ligne.get("variable", "")).strip()
            modalite = str(ligne.get("modalite", "")).strip()
            balise = f"**** *{variable}_{modalite}".rstrip("_")

        segments_concat.append(f"{balise}\n{texte}")

    return "\n\n".join(segments_concat)


def fusionner_textes_par_variable(df_modalites: pd.DataFrame, variable: str) -> str:
    """Concatène tous les textes d'une variable (toutes modalités confondues)."""

    if not variable:
        return ""

    df_variable = filtrer_modalites(df_modalites, modalites=[], variable=variable)
    return fusionner_textes_modalites(df_variable)


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
