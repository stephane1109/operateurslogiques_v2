"""Fonctions d'analyse normalisée pour les corpus IRaMuTeQ."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import altair as alt
import pandas as pd
import streamlit as st

from analyses import render_analyses_tab
from iramuteq.corpusiramuteq import frequences_marqueurs_par_modalite
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
    dico_marqueurs: Optional[Dict[str, str]] = None,
    dico_memoires: Optional[Dict[str, str]] = None,
    dico_consq: Optional[Dict[str, str]] = None,
    dico_causes: Optional[Dict[str, str]] = None,
    dico_tensions: Optional[Dict[str, str]] = None,
) -> None:
    """Affiche l'analyse normalisée (comparative) pour les modalités IRaMuTeQ."""

    st.markdown("### Analyse comparative par variable / modalité (texte normalisé)")

    if df_modalites.empty:
        st.info("Aucune modalité disponible pour la variable sélectionnée.")
        return

    df_variable = df_modalites.copy()
    df_variable["variable"] = df_variable["variable"].fillna(variable_selectionnee)
    df_variable["modalite"] = df_variable["modalite"].fillna("")
    df_variable["texte_normalise"] = (
        df_variable["texte"].fillna("").apply(lambda t: normaliser_espace(str(t)))
    )

    st.markdown("**Normalisation des textes (analyse sur le texte complet)**")

    df_variable["texte_normalise_analyse"] = df_variable["texte_normalise"]

    df_comparatif = (
        df_variable.groupby(["variable", "modalite"], as_index=False)
        .agg(
            {
                "texte_normalise": lambda s: " \n".join([v for v in s.tolist() if str(v).strip()]),
                "texte_normalise_analyse": lambda s: " \n".join(
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
        "(texte complet analysé pour chaque modalité)"
    )
    st.dataframe(df_comparatif, use_container_width=True)

    frequences_modalites = []
    for _, ligne in df_comparatif.iterrows():
        texte_modalite = str(ligne.get("texte_normalise_analyse", ""))
        if not texte_modalite.strip():
            continue

        detections_modalite = preparer_detections_fn(
            texte_modalite,
            use_regex_cc,
            dico_connecteurs=dico_connecteurs_iramuteq,
            dico_marqueurs={},
        )

        freq_modalite = frequences_marqueurs_par_modalite(detections_modalite)
        freq_modalite = filtrer_freq_connecteurs(freq_modalite)
        if freq_modalite.empty:
            continue

        freq_modalite["variable"] = ligne.get("variable", "")
        freq_modalite["modalite"] = ligne.get("modalite", "")
        freq_modalite["variable_modalite"] = freq_modalite.apply(
            lambda r: f"{r['variable']} — {r['modalite']}", axis=1
        )
        frequences_modalites.append(freq_modalite)

    texte_modalites_normalise = normaliser_espace(texte_modalites)

    if frequences_modalites:
        df_freq_comparatif = pd.concat(frequences_modalites, ignore_index=True)
        st.markdown("**Fréquences comparatives des connecteurs logiques (texte normalisé)**")
        st.dataframe(df_freq_comparatif, use_container_width=True)

        chart_barres = (
            alt.Chart(df_freq_comparatif)
            .mark_bar()
            .encode(
                x=alt.X("modalite:N", title="Modalité"),
                y=alt.Y("frequence:Q", title="Fréquence"),
                color=alt.Color("categorie:N", title="Catégorie"),
                column=alt.Column("variable:N", title="Variable"),
                tooltip=["variable", "modalite", "categorie", "frequence"],
            )
            .resolve_scale(y="independent")
        )
        st.altair_chart(chart_barres, use_container_width=True)

        df_freq_par_variable = (
            df_freq_comparatif.groupby(["variable", "categorie"], as_index=False)[
                "frequence"
            ]
            .sum()
            .sort_values(["variable", "categorie"])
        )
        if not df_freq_par_variable.empty:
            st.markdown(
                "**Évolution normalisée des fréquences par variable (comparaison en courbes)**"
            )
            chart_courbes = (
                alt.Chart(df_freq_par_variable)
                .mark_line(point=True)
                .encode(
                    x=alt.X("categorie:N", title="Catégorie"),
                    y=alt.Y("frequence:Q", title="Fréquence normalisée"),
                    color=alt.Color("variable:N", title="Variable"),
                    tooltip=["variable", "categorie", "frequence"],
                )
            )
            st.altair_chart(chart_courbes, use_container_width=True)
    else:
        st.info("Aucun connecteur logique détecté dans les modalités normalisées sélectionnées.")

    render_analyses_tab(
        "Corpus IRaMuTeQ (modalités sélectionnées)",
        texte_modalites_normalise,
        preparer_detections_fn(
            texte_modalites_normalise,
            use_regex_cc,
            dico_connecteurs=dico_connecteurs_iramuteq,
            dico_marqueurs=dico_marqueurs or {},
        ),
        use_regex_cc=use_regex_cc,
        hidden_sections={
            "marqueurs",
            "tensions_semantiques",
            "memoires",
            "regex_consequence",
            "regex_cause",
        },
        dico_connecteurs=dico_connecteurs_iramuteq,
        dico_marqueurs=dico_marqueurs or {},
        dico_memoires=dico_memoires or {},
        dico_consq=dico_consq or {},
        dico_causes=dico_causes or {},
        dico_tensions=dico_tensions or {},
        key_prefix="iramuteq_",
    )
