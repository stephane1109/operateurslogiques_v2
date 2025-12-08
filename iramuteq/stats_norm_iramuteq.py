"""Fonctions d'analyse normalisée pour les corpus IRaMuTeQ."""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd
import streamlit as st

from analyses import render_analyses_tab
from iramuteq.corpusiramuteq import frequences_marqueurs_par_modalite
from stats_norm import render_stats_norm_tab
from text_utils import normaliser_espace


def filtrer_freq_connecteurs(df_freq: pd.DataFrame) -> pd.DataFrame:
    """Ne conserve que les fréquences des connecteurs logiques."""

    if df_freq is None or df_freq.empty or "type" not in df_freq:
        return pd.DataFrame(columns=["type", "categorie", "frequence"])

    freq_connecteurs = df_freq[df_freq["type"] == "Connecteur logique"]
    if freq_connecteurs.empty:
        return pd.DataFrame(columns=["type", "categorie", "frequence"])

    return freq_connecteurs.reset_index(drop=True)


def limiter_nb_mots(texte: str, nb_mots: int) -> str:
    """Retourne le texte limité au nombre de mots souhaité."""

    mots = str(texte).split()
    if nb_mots <= 0 or not mots:
        return ""
    return " ".join(mots[:nb_mots])


def render_normalisation_corpus(
    df_modalites: pd.DataFrame,
    variable_selectionnee: str,
    texte_modalites: str,
    use_regex_cc: bool,
    dico_connecteurs_iramuteq: Dict[str, str],
    preparer_detections_fn: Callable[..., Dict[str, pd.DataFrame]],
) -> None:
    """Affiche l'analyse normalisée (comparative) pour les modalités IRaMuTeQ."""

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
        lambda t: limiter_nb_mots(t, int(nb_mots_normalisation))
    )

    df_comparatif = (
        df_variable.groupby(["variable", "modalite"], as_index=False)
        .agg(
            {
                "texte_normalise": lambda s: " \n".join([v for v in s.tolist() if str(v).strip()]),
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

    texte_modalites_normalise = limiter_nb_mots(
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
    freq_modalite_compare = filtrer_freq_connecteurs(freq_modalite_compare)

    freq_modalites_normalise = frequences_marqueurs_par_modalite(
        detections_modalites_normalises
    )
    freq_modalites_normalise = filtrer_freq_connecteurs(freq_modalites_normalise)

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
        df_memoires_2=detections_modalites_normalises.get("df_memoires", pd.DataFrame()),
        df_consq_lex_2=detections_modalites_normalises.get("df_consq_lex", pd.DataFrame()),
        df_causes_lex_2=detections_modalites_normalises.get("df_causes_lex", pd.DataFrame()),
        df_tensions_2=detections_modalites_normalises.get("df_tensions", pd.DataFrame()),
        heading_discours_1=f"{variable_selectionnee} — {modalite_compare}",
        heading_discours_2=f"{variable_selectionnee} — sélection",
        couleur_discours_1="#c00000",
        couleur_discours_2="#1f4e79",
        sections=["reperes", "connecteurs"],
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
