"""Analyse de sentiments basée sur vaderSentiment-fr pour les discours."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from text_utils import normaliser_espace, segmenter_en_phrases


@dataclass
class SentimentResult:
    """Représentation d'un score de sentiment par phrase."""

    id_phrase: int
    texte: str
    label: str
    score: float
    valence: float


@st.cache_resource(show_spinner=False)
def _charger_pipeline_sentiment():
    """Charge l'analyseur VADER adapté au français."""

    try:
        from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as err:  # pragma: no cover - dépendance manquante
        raise RuntimeError(
            "Impossible d'importer vaderSentiment-fr pour l'analyse de sentiments"
            f" : {err}"
        ) from err

    try:
        return SentimentIntensityAnalyzer()
    except Exception as err:  # pragma: no cover - initialisation improbable
        raise RuntimeError(
            "Impossible d'initialiser l'analyseur vaderSentiment-fr"
            f" : {err}"
        ) from err


def _label_valence_from_score(score: float) -> tuple[str, float]:
    """Retourne le label textuel et la valence (compound) depuis un score VADER."""

    if score >= 0.05:
        return "positive", score
    if score <= -0.05:
        return "negative", score
    return "neutral", score


def analyser_sentiments_discours(
    texte: str,
    pipeline_sentiment,
) -> List[SentimentResult]:
    """Retourne les scores de sentiment pour chaque phrase du discours."""

    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm) if texte_norm else []
    if not phrases:
        return []

    resultats: List[SentimentResult] = []
    for idx, phrase in enumerate(phrases, start=1):
        scores = pipeline_sentiment.polarity_scores(phrase)
        compound = float(scores.get("compound", 0.0))
        label_pred, valence = _label_valence_from_score(compound)
        proba = abs(compound)
        resultats.append(
            SentimentResult(
                id_phrase=idx,
                texte=phrase,
                label=label_pred,
                score=proba,
                valence=valence,
            )
        )
    return resultats


def _construire_dataframe(resultats: List[SentimentResult]) -> pd.DataFrame:
    """Convertit les résultats en DataFrame et ajoute un lissage."""

    if not resultats:
        return pd.DataFrame(
            columns=["id_phrase", "texte", "label", "score", "valence", "valence_lissee"]
        )

    df = pd.DataFrame([r.__dict__ for r in resultats])
    return df


def _ajouter_lissage(df: pd.DataFrame, fenetre: int) -> pd.DataFrame:
    """Calcule la moyenne glissante de la valence."""

    if df.empty:
        return df
    df = df.copy()
    df["valence_lissee"] = df["valence"].rolling(
        window=fenetre, min_periods=1, center=True
    ).mean()
    return df


def _afficher_intro_methodologie():
    """Affiche le texte méthodologique demandé."""

    st.markdown("### ASentsVader")
    st.markdown(
        """
        Analyse de Sentiments avec le dictionnaire VADER

        Analyse lexicale avec vaderSentiment-fr.
        Cette méthode exploite l'adaptation française de VADER (Valence Aware Dictionary for Sentiment Reasoning), basée sur un lexique et des règles pour estimer la valence des phrases.

        Ressource : la librairie Python est disponible sur PyPI : [vaderSentiment-fr](https://pypi.org/project/vaderSentiment-fr/).

        La Courbe de Valence Émotionnelle
        Pour construire ce graph, le script doit attribuer un score à chaque phrase :

            Axe X (Temps) : Le déroulement du discours (du début à la fin).
            Axe Y (Valence) : L'axe émotionnel.
                Haut (+) : Chance, Espoir, Solution, Joie, Force....
                Bas (-) : Malchance, Péril, Problème, Tristesse, Faiblesse....

        Les données brutes (phrase par phrase) donnent un graph avec beaucoup de "bruit". On peut corriger cela avec la moyenne glissante (smoothing) sur 5 ou x phrases pour voir apparaître une courbe "smooth" .

        Visualisation - exemples :

            La Forme "Man in a Hole" (L'Homme dans le trou) — courbe en V ou en U.
            La Forme "Icarus" (La Tragédie) — courbe en V inversé (Montée puis Chute).
            La Forme "Rags to Riches" (L'Ascension / La Rampe).
        """
    )


def _visualiser_courbe(df: pd.DataFrame, titre: str):
    """Affiche la courbe de valence émotionnelle."""

    if df.empty:
        st.info("Aucune phrase à afficher pour ce discours.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("id_phrase:Q", title="Temps (phrases)"),
            y=alt.Y("valence_lissee:Q", title="Valence (moyenne glissante)"),
            tooltip=["id_phrase", "valence", "valence_lissee", "label", "texte"],
        )
        .properties(title=titre)
    )
    st.altair_chart(chart, use_container_width=True)


def render_sentiments_tab(
    texte_discours_1: str,
    texte_discours_2: str,
    nom_discours_1: str,
    nom_discours_2: str,
):
    """Rendu Streamlit pour l'onglet d'analyse de sentiments."""

    _afficher_intro_methodologie()

    if not texte_discours_1.strip() and not texte_discours_2.strip():
        st.info("Aucun discours fourni. Ajoutez du texte pour lancer l'analyse de sentiments.")
        return

    try:
        pipe_sentiment = _charger_pipeline_sentiment()
    except Exception as err:
        st.error(str(err))
        return

    fenetre = st.slider(
        "Taille de la moyenne glissante (en nombre de phrases)",
        min_value=3,
        max_value=50,
        value=5,
        step=1,
    )

    discours_disponibles: Dict[str, str] = {}
    if texte_discours_1.strip():
        discours_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        discours_disponibles[nom_discours_2] = texte_discours_2

    onglets = st.tabs(list(discours_disponibles.keys()))
    for tab, (nom, contenu) in zip(onglets, discours_disponibles.items()):
        with tab:
            st.markdown(f"#### {nom}")
            resultats = analyser_sentiments_discours(contenu, pipe_sentiment)
            df_res = _ajouter_lissage(_construire_dataframe(resultats), fenetre)
            if df_res.empty:
                st.info("Aucune phrase détectée pour ce discours.")
                continue

            st.dataframe(
                df_res[["id_phrase", "texte", "label", "score", "valence", "valence_lissee"]],
                use_container_width=True,
            )
            _visualiser_courbe(df_res, titre=f"Courbe de valence émotionnelle — {nom}")

