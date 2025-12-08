"""Affichage et analyses statistiques pour l'onglet IRaMuTeQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import streamlit as st

from analyses import (
    COULEURS_BADGES,
    COULEURS_MARQUEURS,
    COULEURS_TENSIONS,
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
from stats_norm import render_stats_norm_tab
from text_utils import normaliser_espace


def _charger_connecteurs_iramuteq(dictionnaires_dir: Path) -> Dict[str, str]:
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


def _limiter_nb_mots(texte: str, nb_mots: int) -> str:
    """Retourne le texte limité au nombre de mots souhaité."""

    mots = str(texte).split()
    if nb_mots <= 0 or not mots:
        return ""
    return " ".join(mots[:nb_mots])


def render_corpus_iramuteq_tab(
    df_modalites: pd.DataFrame,
    dictionnaires_dir: Path,
    use_regex_cc: bool,
    preparer_detections_fn: Callable[..., Dict[str, pd.DataFrame]],
    *,
    dico_marqueurs: Dict[str, str],
    dico_memoires: Dict[str, str],
    dico_consq: Dict[str, str],
    dico_causes: Dict[str, str],
    dico_tensions: Dict[str, str],
) -> None:
    """Affiche les analyses statistiques pour les corpus IRaMuTeQ."""

    def _filtrer_freq_connecteurs(df_freq: pd.DataFrame) -> pd.DataFrame:
        """Ne conserve que les fréquences des connecteurs logiques."""

        if df_freq is None or df_freq.empty or "type" not in df_freq:
            return pd.DataFrame(columns=["type", "categorie", "frequence"])

        freq_connecteurs = df_freq[df_freq["type"] == "Connecteur logique"]
        if freq_connecteurs.empty:
            return pd.DataFrame(columns=["type", "categorie", "frequence"])

        return freq_connecteurs.reset_index(drop=True)

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

    st.markdown("### Analyses par modalité sélectionnée")
    for idx, ligne in df_selection.iterrows():
        modalite_courante = str(ligne.get("modalite", "")).strip()
        texte_modalite = str(ligne.get("texte", ""))
        detections_modalite = preparer_detections_fn(
            texte_modalite,
            use_regex_cc,
            dico_connecteurs=dico_connecteurs_iramuteq,
            dico_marqueurs={},
            dico_memoires={},
            dico_consq={},
            dico_causes={},
            dico_tensions={},
        )
        freq_modalite = frequences_marqueurs_par_modalite(detections_modalite)
        freq_modalite = _filtrer_freq_connecteurs(freq_modalite)

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
        dico_memoires={},
        dico_consq={},
        dico_causes={},
        dico_tensions={},
    )
    freq_selection = frequences_marqueurs_par_modalite(detections_modalites)
    freq_selection = _filtrer_freq_connecteurs(freq_selection)
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

    st.markdown("### Analyse comparative par variable / modalité (texte normalisé)")

    df_variable = df_modalites[df_modalites["variable"] == variable_selectionnee]
    if df_variable.empty:
        st.info("Aucune modalité disponible pour la variable sélectionnée.")
        return

    df_variable = df_variable.copy()
    df_variable["texte_normalise"] = (
        df_variable["texte"].fillna("").apply(lambda t: normaliser_espace(str(t)))
    )

    st.markdown("**Normalisation des textes (nombre de mots analysés)**")
    nb_mots_normalisation = st.number_input(
        "Nombre de mots à conserver pour l'analyse comparative (par modalité)",
        min_value=50,
        max_value=5000,
        value=1000,
        step=50,
    )

    df_variable["texte_normalise_limite"] = df_variable["texte_normalise"].apply(
        lambda t: _limiter_nb_mots(t, int(nb_mots_normalisation))
    )

    df_comparatif = (
        df_variable.groupby(["variable", "modalite"], as_index=False)
        .agg(
            {
                "texte_normalise": lambda s: " \n".join(
                    [v for v in s.tolist() if str(v).strip()]
                ),
                "texte_normalise_limite": lambda s: " \n".join(
                    [v for v in s.tolist() if str(v).strip()]
                ),
            }
        )
        .reset_index(drop=True)
    )

    if df_comparatif.empty:
        st.info("Impossible de constituer un tableau comparatif des textes normalisés.")
        return

    st.markdown(
        "**Textes concaténés et normalisés par modalité**\n"
        "(texte tronqué au nombre de mots choisi pour rendre les statistiques comparables)"
    )
    st.dataframe(df_comparatif, use_container_width=True)

    modalite_compare = st.selectbox(
        "Choisir une modalité à analyser avec les opérateurs logiques",
        options=df_comparatif["modalite"].tolist(),
    )

    texte_modalite_compare = " ".join(
        df_comparatif[df_comparatif["modalite"] == modalite_compare][
            "texte_normalise_limite"
        ].tolist()
    )

    texte_modalites_normalise = _limiter_nb_mots(
        normaliser_espace(texte_modalites), int(nb_mots_normalisation)
    )

    st.markdown(
        "**Texte normalisé analysé** (tronqué à "
        f"{int(nb_mots_normalisation)} mot(s) pour harmoniser les statistiques)"
    )
    st.text_area(
        "texte_modalite_compare",
        texte_modalite_compare,
        height=200,
        key=f"txt_modalite_compare_{modalite_compare}",
        label_visibility="collapsed",
    )

    detections_modalite_compare = preparer_detections_fn(
        texte_modalite_compare,
        use_regex_cc,
        dico_connecteurs=dico_connecteurs_iramuteq,
        dico_marqueurs={},
        dico_memoires={},
        dico_consq={},
        dico_causes={},
        dico_tensions={},
    )

    detections_modalites_normalises = preparer_detections_fn(
        texte_modalites_normalise,
        use_regex_cc,
        dico_connecteurs=dico_connecteurs_iramuteq,
        dico_marqueurs={},
        dico_memoires={},
        dico_consq={},
        dico_causes={},
        dico_tensions={},
    )

    freq_modalite_compare = frequences_marqueurs_par_modalite(detections_modalite_compare)
    freq_modalite_compare = _filtrer_freq_connecteurs(freq_modalite_compare)

    freq_modalites_normalise = frequences_marqueurs_par_modalite(
        detections_modalites_normalises
    )
    freq_modalites_normalise = _filtrer_freq_connecteurs(freq_modalites_normalise)

    st.markdown("**Fréquences des connecteurs logiques (texte normalisé)**")
    if freq_modalite_compare.empty:
        st.info("Aucun connecteur logique détecté pour cette modalité normalisée.")
    else:
        st.dataframe(freq_modalite_compare, use_container_width=True)
        st.bar_chart(
            freq_modalite_compare,
            x="categorie",
            y="frequence",
            color="type",
        )

    render_analyses_tab(
        f"Analyse comparative : {variable_selectionnee} / {modalite_compare}",
        texte_modalite_compare,
        detections_modalite_compare,
        use_regex_cc=use_regex_cc,
        hidden_sections={
            "marqueurs",
            "tensions_semantiques",
            "regex_consequence",
            "regex_cause",
            "memoires",
        },
        dico_connecteurs=dico_connecteurs_iramuteq,
        dico_marqueurs={},
        dico_memoires={},
        dico_consq={},
        dico_causes={},
        dico_tensions={},
        key_prefix="iramuteq_compare_",
    )

    st.markdown("**Statistiques normalisées sur le texte normalisé**")
    render_stats_norm_tab(
        texte_modalite_compare,
        detections_modalite_compare.get("df_conn", pd.DataFrame()),
        detections_modalite_compare.get("df_marq", pd.DataFrame()),
        detections_modalite_compare.get("df_memoires", pd.DataFrame()),
        detections_modalite_compare.get("df_consq_lex", pd.DataFrame()),
        detections_modalite_compare.get("df_causes_lex", pd.DataFrame()),
        detections_modalite_compare.get("df_tensions", pd.DataFrame()),
        texte_source_2=texte_modalites_normalise,
        df_conn_2=detections_modalites_normalises.get("df_conn", pd.DataFrame()),
        df_marqueurs_2=detections_modalites_normalises.get("df_marq", pd.DataFrame()),
        df_memoires_2=detections_modalites_normalises.get(
            "df_memoires", pd.DataFrame()
        ),
        df_consq_lex_2=detections_modalites_normalises.get(
            "df_consq_lex", pd.DataFrame()
        ),
        df_causes_lex_2=detections_modalites_normalises.get(
            "df_causes_lex", pd.DataFrame()
        ),
        df_tensions_2=detections_modalites_normalises.get(
            "df_tensions", pd.DataFrame()
        ),
        heading_discours_1=f"{variable_selectionnee} — {modalite_compare}",
        heading_discours_2=f"{variable_selectionnee} — sélection",
        couleur_discours_1="#c00000",
        couleur_discours_2="#1f4e79",
    )

    st.markdown("**Connecteurs logiques — corpus normalisé (sélection)**")
    if freq_modalites_normalise.empty:
        st.info("Aucun connecteur logique détecté dans le corpus normalisé.")
    else:
        st.dataframe(freq_modalites_normalise, use_container_width=True)
        st.bar_chart(
            freq_modalites_normalise,
            x="categorie",
            y="frequence",
            color="type",
        )

    render_analyses_tab(
        "Corpus IRaMuTeQ (modalités sélectionnées)",
        texte_modalites_normalise,
        detections_modalites_normalises,
        use_regex_cc=use_regex_cc,
        hidden_sections={
            "marqueurs",
            "tensions_semantiques",
            "regex_consequence",
            "regex_cause",
            "memoires",
        },
        dico_connecteurs=dico_connecteurs_iramuteq,
        dico_marqueurs={},
        dico_memoires={},
        dico_consq={},
        dico_causes={},
        dico_tensions={},
        key_prefix="iramuteq_",
    )
