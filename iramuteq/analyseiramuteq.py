"""Analyse IRaMuTeQ : statistiques des connecteurs logiques par modalité.

Ce module propose une vue dédiée dans Streamlit pour :
- charger un dictionnaire de connecteurs spécifique à IRaMuTeQ (dictionnaires/connecteursiramuteq.json) ;
- calculer les fréquences des connecteurs par variable/modalité ;
- afficher un histogramme comparatif ;
- surligner les connecteurs dans les textes agrégés.
- cumuler plusieurs variables/modalités pour une analyse transversale.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from analyses import construire_regex_depuis_liste
from iramuteq.corpusiramuteq import filtrer_modalites
from text_utils import normaliser_espace, segmenter_en_phrases


def _charger_connecteurs_iramuteq(dictionnaires_dir: Path) -> Dict[str, str]:
    """Charge le dictionnaire des connecteurs IRaMuTeQ en normalisant les clés."""

    chemin = Path(dictionnaires_dir) / "connecteursiramuteq.json"
    if not chemin.exists():
        st.error(
            "Le fichier 'dictionnaires/connecteursiramuteq.json' est introuvable."
        )
        return {}

    try:
        with chemin.open("r", encoding="utf-8") as f:
            donnees = json.load(f)
    except Exception as exc:  # pragma: no cover - gestion Streamlit
        st.error(f"Impossible de charger le dictionnaire IRaMuTeQ : {exc}")
        return {}

    connecteurs = {str(k).strip(): str(v).strip() for k, v in donnees.items() if k}
    return connecteurs


def _detecter_connecteurs(texte: str, dico_conn: Dict[str, str]) -> pd.DataFrame:
    """Détecte les connecteurs logiques dans un texte en utilisant le dictionnaire fourni."""

    if not texte or not dico_conn:
        return pd.DataFrame(columns=["id_phrase", "phrase", "connecteur", "code", "position", "longueur"])

    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm)
    motifs = construire_regex_depuis_liste(list(dico_conn.keys()))

    occurrences: List[dict] = []
    for i, ph in enumerate(phrases, start=1):
        ph_norm = normaliser_espace(ph)
        for cle_norm, motif in motifs:
            for m in motif.finditer(ph_norm):
                occurrences.append(
                    {
                        "id_phrase": i,
                        "phrase": ph_norm,
                        "connecteur": cle_norm,
                        "code": dico_conn.get(cle_norm, ""),
                        "position": m.start(),
                        "longueur": m.end() - m.start(),
                    }
                )

    df = pd.DataFrame(occurrences)
    if not df.empty:
        df.sort_values(by=["id_phrase", "position", "longueur"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    return df


def _annoter_phrase(phrase: str, occs: Iterable[dict]) -> str:
    """Retourne le HTML d'une phrase avec connecteurs surlignés."""

    if not phrase:
        return ""

    morceaux: List[str] = []
    curseur = 0
    for occ in occs:
        debut = int(occ.get("position", 0))
        fin = debut + int(occ.get("longueur", 0))
        if debut < curseur:
            continue
        morceaux.append(html.escape(phrase[curseur:debut]))
        extrait = html.escape(phrase[debut:fin])
        code = html.escape(str(occ.get("code", "")).upper())
        morceaux.append(
            f"<span class='iramuteq-conn'>{extrait}<span class='iramuteq-code'>{code}</span></span>"
        )
        curseur = fin
    morceaux.append(html.escape(phrase[curseur:]))
    return "".join(morceaux)


def _annoter_texte(df_conn: pd.DataFrame) -> str:
    """Construit un bloc HTML annoté pour l'ensemble des phrases détectées."""

    if df_conn is None or df_conn.empty:
        return "<div class='iramuteq-texte'>(Aucun connecteur détecté)</div>"

    blocs: List[str] = [
        "<style>",
        ".iramuteq-texte { line-height: 1.7; font-size: 1rem; white-space: pre-wrap; }",
        ".iramuteq-conn { background: #fff6e5; padding: 0.05rem 0.15rem; border-radius: 0.2rem; border: 1px solid #f0b44c; font-weight: 600; }",
        ".iramuteq-code { margin-left: 0.25rem; font-size: 0.8em; color: #c05621; font-family: monospace; }",
        "</style>",
        "<div class='iramuteq-texte'>",
    ]

    for id_phrase, occs in df_conn.groupby("id_phrase"):
        phrase = occs.iloc[0]["phrase"]
        blocs.append(_annoter_phrase(str(phrase), occs.to_dict("records")))
    blocs.append("</div>")
    return "".join(blocs)


def _statistiques_par_modalite(detections: pd.DataFrame) -> pd.DataFrame:
    """Calcule la fréquence des connecteurs par modalité."""

    if detections is None or detections.empty:
        return pd.DataFrame(columns=["modalite", "code", "frequence"])

    freq = (
        detections.groupby(["modalite", "code"])
        .size()
        .reset_index(name="frequence")
        .sort_values(by=["modalite", "frequence"], ascending=[True, False])
    )
    return freq


def render_corpus_iramuteq_tab(
    df_modalites: pd.DataFrame,
    dictionnaires_dir: Path,
    use_regex_cc: bool,
    preparer_detections,
):
    """Affiche l'analyse statistique des connecteurs IRaMuTeQ dans Streamlit."""

    st.subheader("Analyse IRaMuTeQ des connecteurs logiques")

    if df_modalites is None or df_modalites.empty:
        st.info("Importez un corpus IRaMuTeQ pour afficher les statistiques.")
        return

    dico_connecteurs = _charger_connecteurs_iramuteq(dictionnaires_dir)
    if not dico_connecteurs:
        return

    variables = sorted({v for v in df_modalites["variable"].dropna() if v})
    if not variables:
        st.warning("Aucune variable détectée dans le corpus.")
        return

    variables_selectionnees = st.multiselect(
        "Variables à analyser",
        variables,
        default=variables[:1],
    )

    if not variables_selectionnees:
        st.warning("Sélectionnez au moins une variable pour lancer l'analyse.")
        return

    st.caption(
        "Choisissez librement plusieurs variables et modalités : un cumul regroupera l'ensemble des sélections."
    )

    modalites_par_variable: Dict[str, List[str]] = {}
    for variable in variables_selectionnees:
        df_var = df_modalites[df_modalites["variable"] == variable]
        modalites_disponibles = sorted({m for m in df_var["modalite"].dropna() if m})
        modalites_selection = st.multiselect(
            f"Modalités pour {variable}",
            modalites_disponibles,
            default=modalites_disponibles,
        )
        modalites_par_variable[variable] = modalites_selection

    groupes_selectionnes: List[str] = []
    textes_cumules: List[Tuple[str, str, str]] = []
    detections_modalites: List[pd.DataFrame] = []

    for variable, modalites_selectionnees in modalites_par_variable.items():
        df_filtre = filtrer_modalites(
            df_modalites, modalites_selectionnees, variable=variable
        )
        if df_filtre.empty:
            st.warning(
                f"Aucune modalité sélectionnée avec du texte pour la variable '{variable}'."
            )
            continue

        for modalite in modalites_selectionnees:
            df_mod = df_filtre[df_filtre["modalite"] == modalite]
            if df_mod.empty:
                continue
            texte_concat = "\n\n".join(df_mod["texte"].astype(str))
            balise_label = f"*{variable}_{modalite}".rstrip("_")
            textes_cumules.append((balise_label, variable, texte_concat))
            df_conn = _detecter_connecteurs(texte_concat, dico_connecteurs)
            if not df_conn.empty:
                label_modalite = f"{variable} • {modalite}" if variable else modalite
                df_conn = df_conn.assign(
                    variable=variable,
                    modalite=label_modalite,
                    modalite_source=modalite,
                )
                if label_modalite not in groupes_selectionnes:
                    groupes_selectionnes.append(label_modalite)
            detections_modalites.append(df_conn)

    ajouter_cumul = st.checkbox(
        "Ajouter un cumul des variables/modalités sélectionnées",
        value=True,
    )

    label_cumul = "Cumul des sélections"
    balises_cumul: List[str] = [b for b, _, _ in textes_cumules if b]

    if balises_cumul:
        label_cumul = "Cumul : " + ", ".join(balises_cumul)

    if ajouter_cumul and textes_cumules:
        segments_cumules = []
        for balise_label, variable, texte in textes_cumules:
            if texte:
                prefix = balise_label or variable or "Sélection"
                segments_cumules.append(f"{prefix}\n{texte}")
        texte_total = "\n\n".join(segments_cumules)
        df_conn_cumul = _detecter_connecteurs(texte_total, dico_connecteurs)
        if not df_conn_cumul.empty:
            df_conn_cumul = df_conn_cumul.assign(
                variable="CUMUL",
                modalite=label_cumul,
                modalite_source="",
            )
            detections_modalites.append(df_conn_cumul)
            if label_cumul not in groupes_selectionnes:
                groupes_selectionnes.append(label_cumul)

    df_detections = (
        pd.concat([d for d in detections_modalites if d is not None], ignore_index=True)
        if detections_modalites
        else pd.DataFrame()
    )

    st.markdown(
        "Les détections ci-dessous utilisent le dictionnaire `connecteursiramuteq.json`."
    )

    if df_detections.empty:
        st.info("Aucun connecteur logique n'a été détecté dans les modalités sélectionnées.")
        return

    df_stats = _statistiques_par_modalite(df_detections)
    if not df_stats.empty:
        df_stats = df_stats[df_stats["modalite"] != label_cumul]

    if not df_stats.empty:
        st.markdown("#### Histogramme comparatif des connecteurs")
        chart = (
            alt.Chart(df_stats)
            .mark_bar()
            .encode(
                x=alt.X("code:N", title="Connecteur logique"),
                y=alt.Y("frequence:Q", title="Fréquence"),
                color=alt.Color("modalite:N", title="Modalité"),
                tooltip=["modalite", "code", "frequence"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df_stats, use_container_width=True)

    st.markdown("#### Connecteurs surlignés dans les textes")
    for modalite in groupes_selectionnes:
        df_mod_dets = df_detections[df_detections["modalite"] == modalite]
        if df_mod_dets.empty:
            continue
        st.markdown(f"**Modalité : {modalite}**")
        st.markdown(_annoter_texte(df_mod_dets), unsafe_allow_html=True)
