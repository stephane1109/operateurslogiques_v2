"""Affichage et analyses statistiques pour l'onglet IRaMuTeQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from analyses import (
    COULEURS_BADGES,
    COULEURS_MARQUEURS,
    css_badges,
    html_annote,
    render_analyses_tab,
)
from iramuteq.corpusiramuteq import (
    filtrer_modalites,
    fusionner_textes_modalites,
    fusionner_textes_par_variable,
    frequences_marqueurs_par_modalite,
)
from iramuteq.stats_norm_iramuteq import (
    filtrer_freq_connecteurs,
    limiter_nb_mots,
    render_normalisation_corpus,
)
from stats_norm import _render_stats_norm_block, render_stats_norm_tab
from text_utils import normaliser_espace


def _charger_connecteurs_iramuteq(dictionnaires_dir: Path) -> dict[str, str]:
    """Charge le dictionnaire spécifique aux analyses IRaMuTeQ."""

    chemin = dictionnaires_dir / "connecteursiramuteq.json"
    if not chemin.is_file():
        raise FileNotFoundError(
            "Dictionnaire 'connecteursiramuteq.json' introuvable dans le dossier 'dictionnaires/'."
        )

    try:
        with chemin.open("r", encoding="utf-8") as f:
            contenu = f.read()
        try:
            data = json.loads(contenu)
        except json.JSONDecodeError:
            contenu_nettoye = contenu.lstrip("\ufeff").replace("{{", "{", 1)
            data = json.loads(contenu_nettoye)
    except Exception as err:  # pragma: no cover - affichage Streamlit
        raise ValueError(f"Impossible de charger le dictionnaire IRaMuTeQ : {err}") from err

    if not isinstance(data, dict):
        raise ValueError(
            "Format JSON non supporté pour 'connecteursiramuteq.json' (dict attendu)."
        )

    return {
        normaliser_espace(k.lower()): str(v).upper()
        for k, v in data.items()
        if k and str(k).strip()
    }


def render_corpus_iramuteq_tab(
    df_modalites: pd.DataFrame,
    dictionnaires_dir: Path,
    use_regex_cc: bool,
    preparer_detections_fn: Callable[..., dict[str, pd.DataFrame]],
) -> None:
    """Affiche les analyses statistiques pour les corpus IRaMuTeQ."""

    st.subheader("Corpus IRaMuTeQ : sélection des modalités")

    if df_modalites.empty:
        st.info("Aucun corpus IRaMuTeQ n'a été importé pour le moment.")
        return

    try:
        dico_connecteurs_iramuteq = _charger_connecteurs_iramuteq(dictionnaires_dir)
    except Exception as err:  # pragma: no cover - affichage Streamlit
        st.error(str(err))
        return

    df_modalites = df_modalites.copy()
    df_modalites["variable"] = df_modalites["variable"].fillna("").replace("", "Corpus")

    variables_disponibles = sorted(
        [v for v in df_modalites["variable"].dropna().unique().tolist() if str(v).strip()]
    )
    if not variables_disponibles:
        variables_disponibles = ["Corpus"]

    options_variables = ["(Choisir une variable)"] + variables_disponibles
    variable_selectionnee = st.selectbox(
        "Choisir une variable",
        options=options_variables,
        index=0,
    )

    if variable_selectionnee == "(Choisir une variable)":
        st.info("Sélectionnez une variable pour afficher les modalités disponibles.")
        return

    modalites_disponibles = sorted(
        [
            m
            for m in df_modalites[df_modalites["variable"] == variable_selectionnee]["modalite"]
            .dropna()
            .unique()
            .tolist()
            if str(m).strip()
        ]
    )

    selection_modalites = st.multiselect(
        "Choisir une ou plusieurs modalités à analyser",
        options=modalites_disponibles,
        default=[],
    )

    if not selection_modalites:
        st.warning("Aucune modalité sélectionnée pour l'analyse.")
        return

    st.markdown(f"**Variable sélectionnée :** {variable_selectionnee}  ")
    st.markdown(
        "**Modalités sélectionnées :** "
        + (", ".join(selection_modalites) if selection_modalites else "Aucune")
    )

    df_selection = filtrer_modalites(df_modalites, selection_modalites, variable_selectionnee)
    if df_selection.empty:
        st.warning("Aucune modalité sélectionnée ou correspondance vide.")
        return

    st.markdown("**Aperçu des textes sélectionnés**")
    st.dataframe(df_selection, use_container_width=True)

    st.markdown("### Statistiques normalisées (par 1 000 mots)")
    st.caption(
        "Calcul basé sur les connecteurs logiques du dictionnaire IRaMuTeQ pour les modalités sélectionnées."
    )

    blocs_stats = []
    for idx, ligne in df_selection.iterrows():
        modalite_courante = str(ligne.get("modalite", "")).strip()
        texte_modalite = str(ligne.get("texte", ""))
        detections_stats = preparer_detections_fn(
            texte_modalite,
            use_regex_cc,
            dico_connecteurs=dico_connecteurs_iramuteq,
            dico_marqueurs={},
        )
        detections_stats = {"df_conn": detections_stats.get("df_conn", pd.DataFrame())}
        blocs_stats.append(
            {
                "heading": f"{variable_selectionnee} — {modalite_courante}",
                "texte": texte_modalite,
                "df_conn": detections_stats.get("df_conn", pd.DataFrame()),
            }
        )

    if not blocs_stats:
        st.info("Aucune modalité disponible pour afficher des statistiques normalisées.")
        return

    couleurs_stats = ["#c00000", "#1f4e79"]
    for bloc_idx, bloc in enumerate(blocs_stats):
        _render_stats_norm_block(
            bloc["texte"],
            bloc["df_conn"],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            heading=bloc["heading"],
            heading_color=couleurs_stats[bloc_idx % len(couleurs_stats)],
            sections=["reperes", "connecteurs"],
        )

    st.markdown("### Analyses par modalité sélectionnée")
    for idx, ligne in df_selection.iterrows():
        modalite_courante = str(ligne.get("modalite", "")).strip()
        texte_modalite = str(ligne.get("texte", ""))
        detections_modalite = preparer_detections_fn(
            texte_modalite,
            use_regex_cc,
            dico_connecteurs=dico_connecteurs_iramuteq,
            dico_marqueurs={},
        )
        detections_modalite = {"df_conn": detections_modalite.get("df_conn", pd.DataFrame())}
        freq_modalite = frequences_marqueurs_par_modalite(detections_modalite)
        freq_modalite = filtrer_freq_connecteurs(freq_modalite)

        with st.expander(f"Analyse : {modalite_courante}", expanded=False):
            st.markdown("**Texte de la modalité**")
            st.text_area(
                "texte_modalite",
                texte_modalite,
                height=200,
                key=f"txt_modalite_{idx}_{modalite_courante}",
                label_visibility="collapsed",
            )

            st.markdown("**Détections et texte annoté**")

            st.markdown("**Paramètres de rendu**")
            st.caption("Les options ci-dessous influent sur le rendu HTML (badges).")
            show_codes = st.checkbox(
                "Afficher les codes des connecteurs (cause, conséquence, obligation…)",
                True,
                key=f"show_codes_{idx}_{modalite_courante}",
            )

            st.markdown("**Texte annoté (HTML)**")
            codes_disponibles = {str(v).upper() for v in dico_connecteurs_iramuteq.values()}

            show_codes_dict = {code: show_codes for code in codes_disponibles}
            show_consequences = use_regex_cc and show_codes
            show_causes = use_regex_cc and show_codes
            texte_annote = html_annote(
                texte_modalite,
                dico_connecteurs_iramuteq,
                {},
                {},
                {},
                {},
                {},
                show_codes_dict,
                show_consequences,
                show_causes,
                True,
                show_marqueurs_categories=None,
                show_memoires_categories=None,
                show_tensions_categories=None,
            )
            st.markdown(texte_annote, unsafe_allow_html=True)

            st.markdown("**Fréquences des marqueurs logiques**")
            if freq_modalite.empty:
                st.info("Aucun marqueur logique détecté pour cette modalité.")
            else:
                st.dataframe(freq_modalite, use_container_width=True)
                st.bar_chart(
                    freq_modalite,
                    x="categorie",
                    y="frequence",
                    color="type",
                )

    if set(selection_modalites) == set(modalites_disponibles):
        texte_modalites = fusionner_textes_par_variable(df_modalites, variable_selectionnee)
    else:
        texte_modalites = fusionner_textes_modalites(df_selection)
    st.markdown("### Texte combiné des modalités sélectionnées")
    st.markdown(
        f"Longueur du texte combiné : {len(texte_modalites)} caractères pour {len(df_selection)} segment(s)"
    )

    detections_modalites = preparer_detections_fn(
        texte_modalites,
        use_regex_cc,
        dico_connecteurs=dico_connecteurs_iramuteq,
        dico_marqueurs={},
    )
    detections_modalites = {"df_conn": detections_modalites.get("df_conn", pd.DataFrame())}
    freq_selection = frequences_marqueurs_par_modalite(detections_modalites)
    freq_selection = filtrer_freq_connecteurs(freq_selection)
    st.markdown("**Fréquences globales des marqueurs (sélection)**")
    if freq_selection.empty:
        st.info("Aucun marqueur logique détecté dans la sélection.")
    else:
        st.dataframe(freq_selection, use_container_width=True)
        st.bar_chart(
            freq_selection,
            x="categorie",
            y="frequence",
            color="type",
        )

    render_normalisation_corpus(
        df_modalites,
        variable_selectionnee,
        texte_modalites,
        use_regex_cc,
        dico_connecteurs_iramuteq,
        preparer_detections_fn,
    )
