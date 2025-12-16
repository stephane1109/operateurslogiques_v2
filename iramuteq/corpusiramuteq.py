"""Outils de lecture et de segmentation de corpus IRaMuTeQ."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pandas as pd


def nettoyer_nom_modalite(modalite_brute: str) -> str:
    """Nettoie une modalité en supprimant les marqueurs et espaces superflus."""
    if modalite_brute is None:
        return ""
    return modalite_brute.strip().strip("*_ $")


def extraire_variable_et_modalite(nom_balise: str) -> Dict[str, str]:
    """Retourne la variable et la modalité à partir d'une balise IRaMuTeQ."""

    nom_nettoye = nettoyer_nom_modalite(nom_balise)
    if not nom_nettoye:
        return {"variable": "", "modalite": ""}

    if "_" in nom_nettoye:
        variable, modalite = nom_nettoye.split("_", 1)
    else:
        variable, modalite = "", nom_nettoye

    return {"variable": variable.strip(), "modalite": modalite.strip()}


def segmenter_corpus_par_modalite(texte_corpus: str) -> pd.DataFrame:
    """Segmente un corpus IRaMuTeQ en DataFrame variable / modalité / texte.

    Le format attendu repose sur des blocs introduits par une ligne "****" (avec
    ou sans balises sur la même ligne) suivie de balises de modalité commençant
    par "*" ou "$" (par exemple "*variable_modalite"). Chaque balise
    rencontrée pour un bloc est associée au texte qui suit jusqu'au prochain
    séparateur "****".
    """

    def _extraire_balises_ligne(ligne: str) -> List[str]:
        if not ligne:
            return []
        tokens = re.findall(r"[\*\$][^\s]+", ligne.strip())
        return [tok for tok in tokens if not tok.startswith("****")]

    def _ajouter_segments(segments: List[dict], balises: List[str], contenu: List[str]):
        texte = "\n".join(contenu).strip()
        if not texte:
            return

        if not balises:
            segments.append(
                {"variable": "", "modalite": "", "texte": texte, "balise": ""}
            )
            return

        for balise in balises:
            infos = extraire_variable_et_modalite(balise)
            variable = infos.get("variable", "")
            modalite = infos.get("modalite", "")
            if not variable and modalite.lower() == "model":
                continue
            if variable or modalite:
                segments.append(
                    {
                        "variable": variable,
                        "modalite": modalite,
                        "texte": texte,
                        "balise": balise.strip(),
                    }
                )

    if not texte_corpus:
        return pd.DataFrame(columns=["variable", "modalite", "texte", "balise"])

    segments: List[dict] = []
    balises_courantes: List[str] = []
    contenu_courant: List[str] = []

    for ligne in texte_corpus.splitlines():
        ligne_strip = ligne.strip()

        if ligne_strip.startswith("****"):
            _ajouter_segments(segments, balises_courantes, contenu_courant)
            balises_courantes = _extraire_balises_ligne(ligne_strip)
            contenu_courant = []
            continue

        if ligne_strip.startswith("*") or ligne_strip.startswith("$"):
            balises_courantes.extend(_extraire_balises_ligne(ligne_strip))
            continue

        contenu_courant.append(ligne)

    _ajouter_segments(segments, balises_courantes, contenu_courant)

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
    ]

    tableaux = [t for t in tableaux if not t.empty]
    if not tableaux:
        return pd.DataFrame(columns=["type", "categorie", "frequence"])

    df_freq = pd.concat(tableaux, ignore_index=True)
    df_freq = df_freq[["type", "categorie", "frequence"]]
    return df_freq.sort_values(by=["type", "categorie"]).reset_index(drop=True)
