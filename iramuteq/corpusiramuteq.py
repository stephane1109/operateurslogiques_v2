"""Outils de lecture et de segmentation de corpus IRaMuTeQ."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def lire_fichier_iramuteq(uploaded_file) -> str:
    """Lit un fichier texte ou une archive .iramuteq et retourne son contenu.

    La fonction tente d'abord d'extraire un fichier .txt depuis une archive
    IRaMuTeQ (fichier ZIP). En cas d'échec ou si le fichier n'est pas une
    archive, elle essaie plusieurs encodages texte courants.
    """

    if uploaded_file is None:
        return ""

    donnees: bytes

    if isinstance(uploaded_file, (str, Path)):
        try:
            donnees = Path(uploaded_file).read_bytes()
        except Exception:
            return ""
    elif hasattr(uploaded_file, "getvalue"):
        donnees = uploaded_file.getvalue()
    elif isinstance(uploaded_file, (bytes, bytearray)):
        donnees = bytes(uploaded_file)
    else:
        try:
            donnees = uploaded_file.read()
        except Exception:
            return ""

    try:
        with zipfile.ZipFile(BytesIO(donnees)) as archive:
            noms_txt = [nom for nom in archive.namelist() if nom.lower().endswith(".txt")]
            if noms_txt:
                with archive.open(noms_txt[0]) as fichier_txt:
                    return fichier_txt.read().decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        pass
    except Exception:
        # On ignore silencieusement l'erreur pour tester ensuite les encodages texte
        pass

    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return donnees.decode(enc)
        except Exception:
            continue

    return donnees.decode("utf-8", errors="ignore")


def charger_corpus_iramuteq(uploaded_file) -> pd.DataFrame:
    """Lit un fichier IRaMuTeQ et retourne son découpage en DataFrame.

    Cette fonction mutualise la lecture (texte ou archive .iramuteq) et la
    segmentation en variables/modalités pour éviter de dupliquer la logique dans
    l'interface Streamlit.
    """

    texte = lire_fichier_iramuteq(uploaded_file)
    if not texte.strip():
        return pd.DataFrame(columns=["variable", "modalite", "texte", "balise"])

    return segmenter_corpus_par_modalite(texte)


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


def fusionner_textes_par_variable(
    df_modalites: pd.DataFrame,
    variable: str,
    modalite: Optional[str] = None,
) -> str:
    """Concatène les textes d'une variable et éventuellement d'une modalité donnée.

    - Si ``modalite`` est fourni, seules les lignes correspondant au couple
      variable/modalité sont conservées.
    - Si ``variable`` suit le format ``"variable_modalite"`` (par ex. ``"modele_gpt"``),
      la partie après le premier underscore est automatiquement utilisée comme
      modalité pour éviter de concaténer les textes d'autres valeurs.
    - Lorsque l'on cible explicitement une modalité, tous les textes associés à ce
      couple variable/modalité sont agrégés dans un seul bloc.
    """

    if not variable:
        return ""

    variable_filtre = variable.strip()
    modalites_filtre: List[str] = []

    if modalite:
        modalite_norm = modalite.strip()
        if modalite_norm:
            modalites_filtre.append(modalite_norm)
    elif "_" in variable_filtre:
        infos = extraire_variable_et_modalite(variable_filtre)
        variable_filtre = infos.get("variable") or variable_filtre
        modalite_extraite = infos.get("modalite", "")
        if modalite_extraite:
            modalites_filtre.append(modalite_extraite)

    df_variable = filtrer_modalites(
        df_modalites,
        modalites=modalites_filtre,
        variable=variable_filtre,
    )

    if modalites_filtre and not df_variable.empty:
        segments_agreges = []
        for (var, mod), df_group in df_variable.groupby(["variable", "modalite"], dropna=False):
            textes = [str(t).strip() for t in df_group.get("texte", []) if str(t).strip()]
            if not textes:
                continue

            balise = next((str(b).strip() for b in df_group.get("balise", []) if str(b).strip()), "")
            if not balise:
                balise = f"**** *{var}_{mod}".rstrip("_")

            segments_agreges.append(
                {
                    "variable": var,
                    "modalite": mod,
                    "texte": "\n\n".join(textes),
                    "balise": balise,
                }
            )

        if segments_agreges:
            df_variable = pd.DataFrame(segments_agreges)

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
