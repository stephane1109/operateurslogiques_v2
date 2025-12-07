"""Onglet d'analyse CamemBERT (classification de sentiments) pour l'application Streamlit."""
from __future__ import annotations

from typing import Dict, List

import html

import pandas as pd
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, CamembertTokenizer, pipeline

from text_utils import normaliser_espace, segmenter_en_phrases


STAR_TO_VALENCE = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

VALEUR_BADGES = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


@st.cache_resource(show_spinner=False)
def _charger_camembert_pipeline():
    """Charge la pipeline CamemBERT pour la classification de sentiments."""

    try:
        try:
            tokenizer = CamembertTokenizer.from_pretrained(
                "cmarkea/distilcamembert-base-sentiment"
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                "cmarkea/distilcamembert-base-sentiment", use_fast=False
            )
        return pipeline(
            "text-classification",
            model="cmarkea/distilcamembert-base-sentiment",
            tokenizer=tokenizer,
        )
    except Exception as exc:  # pragma: no cover - uniquement déclenché en environnement Streamlit
        st.error(
            "Impossible de charger CamemBERT (sentiment). Vérifiez la connexion et les dépendances nécessaires."
        )
        st.exception(exc)
        return None


def _construire_df_sentiments(phrases: List[str], predictions) -> pd.DataFrame:
    """Convertit les scores du modèle en DataFrame lisible par phrase."""

    if not predictions:
        return pd.DataFrame(
            columns=["id_phrase", "texte_phrase", "valence", "score_valence"]
        )

    lignes = []
    for idx, (phrase, scores) in enumerate(zip(phrases, predictions), start=1):
        if not scores:
            continue

        scores_valence = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for score in scores:
            etiquette = score.get("label", "")
            valence = STAR_TO_VALENCE.get(etiquette.lower())
            if valence:
                scores_valence[valence] += score.get("score", 0)

        meilleure_valence = max(scores_valence, key=scores_valence.get)
        ligne = {
            "id_phrase": idx,
            "texte_phrase": phrase,
            "valence": meilleure_valence,
            "score_valence": scores_valence[meilleure_valence],
        }

        for nom_valence, val in scores_valence.items():
            ligne[f"score_{nom_valence}"] = val

        for score in scores:
            etiquette = score.get("label", "").lower().replace(" ", "_")
            ligne[f"score_{etiquette}"] = score.get("score", 0)

        lignes.append(ligne)

    return pd.DataFrame(lignes)


def _tracer_barres_scores(df_sentiments: pd.DataFrame):
    """Affiche un graphique Altair des scores par phrase."""

    if df_sentiments.empty:
        st.info("Aucune phrase à représenter.")
        return

    df_barres = df_sentiments.copy()
    chart = (
        alt.Chart(df_barres)
        .mark_bar()
        .encode(
            x=alt.X("id_phrase:O", title="Phrase"),
            y=alt.Y("score_valence:Q", title="Score agrégé (valence)"),
            color=alt.Color(
                "valence:N",
                title="Valence",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["seagreen", "lightgray", "indianred"],
                ),
            ),
            tooltip=[
                "id_phrase",
                "valence",
                alt.Tooltip("score_valence:Q", format=".3f"),
            ],
        )
        .properties(height=300, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _ajouter_lissage(df_sentiments: pd.DataFrame, fenetre: int) -> pd.DataFrame:
    """Calcule une moyenne glissante de la valence pondérée par le score."""

    if df_sentiments.empty:
        return df_sentiments

    poids_valence = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    df_lisse = df_sentiments.copy()
    df_lisse = df_lisse.sort_values("id_phrase")
    df_lisse["valence_ponderee"] = (
        df_lisse["valence"].str.lower().map(poids_valence).fillna(0)
        * df_lisse["score_valence"].astype(float)
    )
    df_lisse["valence_lissee"] = df_lisse["valence_ponderee"].rolling(
        window=fenetre, min_periods=1, center=True
    ).mean()
    return df_lisse


def _tracer_courbe_valence(df_sentiments: pd.DataFrame, fenetre: int):
    """Affiche une courbe de valence similaire à celle de l'analyse VADER."""

    if df_sentiments.empty:
        st.info("Aucune phrase à représenter.")
        return

    df_lisse = _ajouter_lissage(df_sentiments, fenetre)
    chart = (
        alt.Chart(df_lisse)
        .mark_line(point=True)
        .encode(
            x=alt.X("id_phrase:Q", title="Temps (phrases)"),
            y=alt.Y("valence_lissee:Q", title="Valence pondérée (moyenne glissante)"),
            tooltip=[
                "id_phrase",
                "valence",
                alt.Tooltip("score_valence:Q", format=".3f"),
                alt.Tooltip("valence_lissee:Q", format=".3f"),
                "texte_phrase",
            ],
        )
        .properties(height=300, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _css_camembert_annotations() -> str:
    return """
<style>
    .cam-texte-annote {
        line-height: 1.7;
        font-size: 1.02rem;
        white-space: pre-wrap;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .cam-annotation {
        display: inline-flex;
        align-items: flex-start;
        padding: 0.5rem 0.65rem;
        margin: 0.1rem 0;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        color: #111;
    }
    .cam-phrase { margin-left: 0.4rem; display: inline-block; color: #111; }
    .cam-badge-valence { font-weight: 700; font-size: 0.92rem; }
    .cam-valence-positive { background-color: #e9f7ef; border-color: #7cc78d; }
    .cam-valence-neutral { background-color: #f6f6f6; border-color: #cfcfcf; }
    .cam-valence-negative { background-color: #fde8e7; border-color: #f09a96; }
</style>
"""


def _texte_annote_camembert(df_sentiments: pd.DataFrame) -> str:
    if df_sentiments.empty:
        return "<div class='cam-texte-annote'>(Aucune phrase à afficher)</div>"

    lignes = []
    for ligne in df_sentiments.itertuples():
        valence = str(ligne.valence).lower()
        badge = VALEUR_BADGES.get(valence, "🔎")
        classe_valence = f"cam-valence-{valence}"
        texte_phrase = html.escape(str(ligne.texte_phrase))
        lignes.append(
            (
                f"<div class='cam-annotation {classe_valence}'>"
                f"<span class='cam-badge-valence'>{badge} {html.escape(valence)}</span>"
                f"<span class='cam-phrase'>{texte_phrase}</span>"
                "</div>"
            )
        )

    return "<div class='cam-texte-annote'>" + "".join(lignes) + "</div>"


def _tracer_moyennes(df_sentiments: pd.DataFrame):
    """Affiche un graphique des moyennes des scores par sentiment."""

    colonnes_scores = [
        col for col in df_sentiments.columns if col in {"score_positive", "score_neutral", "score_negative"}
    ]
    if not colonnes_scores:
        return

    df_moyennes = (
        df_sentiments[colonnes_scores]
        .mean()
        .reset_index()
        .rename(columns={"index": "sentiment", 0: "score"})
    )
    df_moyennes["sentiment"] = df_moyennes["sentiment"].str.replace("score_", "")

    chart = (
        alt.Chart(df_moyennes)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("score:Q", title="Score moyen agrégé"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["seagreen", "lightgray", "indianred"],
                ),
            ),
            tooltip=["sentiment", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=250, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _selectionner_texte(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
) -> str:
    """Offre une sélection rapide entre les deux discours et une zone d'édition."""

    textes_disponibles: Dict[str, str] = {}
    if texte_discours_1.strip():
        textes_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        textes_disponibles[nom_discours_2] = texte_discours_2

    choix = None
    if textes_disponibles:
        choix = st.selectbox(
            "Choisissez un discours à charger dans la zone de test",
            options=list(textes_disponibles.keys()),
            help="Le texte sélectionné est pré-rempli ci-dessous pour l'inférence CamemBERT.",
        )

    contenu_initial = textes_disponibles.get(choix, "") if choix else ""
    return st.text_area(
        "Texte à analyser",
        value=contenu_initial
        or "C'est formidable de voir tout le monde aujourd'hui pour échanger ensemble !",
        height=200,
    )


def render_camembert_tab(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
):
    """Rendu Streamlit pour l'onglet AnalysSentCamemBert."""

    st.markdown("### AnalysSentCamemBert")
    st.caption(
        """
        **Modèle :** [cmarkea/distilcamembert-base-sentiment](https://huggingface.co/cmarkea/distilcamembert-base-sentiment)

        **Jeux de données d'entraînement (français) :**
        - Environ 205 000 avis clients Amazon (dataset `amazon_reviews_multi`, version française)
        - Environ 235 000 critiques de films du site AlloCiné (dataset `tblard/allocine`)

        Les étiquettes d'origine (1 à 5 étoiles) sont regroupées en valence positive / neutre / négative.
        Le modèle est adapté à l'analyse de phrases ou de commentaires courts en français.
        """
    )

    st.markdown(
        """
        Phrase exemple : "Mesdames et messieurs les parlementaires, il faut savoir tirer les bienfaits d'une crise."\
        Approche par intelligence artificielle (CamemBERT) contre l'approche "dictionnaire" (VADER).\
        Différence d'interprétation :
        * VADER (Dictionnaire) : le mot "crise" ➡️ Négatif.
        * CamemBERT (Contexte) : A lu la phrase entière ("tirer les bienfaits") ➡️ Positif (0.78).
        """
    )

    st.markdown(
        """
        **Comment fonctionne cette analyse ?**

        * Le modèle [CamemBERT](https://huggingface.co/cmarkea/distilcamembert-base-sentiment) est spécialisé pour le français.
        * Chaque discours est découpé en phrases avant d'être envoyé au modèle de classification.
        * Les étiquettes « 1 à 5 étoiles » sont converties en trois sentiments (positif, neutre, négatif) puis agrégées pour donner un score par phrase.
        * Les tableaux et graphiques ci-dessous affichent ces scores pour visualiser la polarité générale du texte.
        """
    )

    texte_cible = _selectionner_texte(texte_discours_1, texte_discours_2, nom_discours_1, nom_discours_2)
    texte_cible = normaliser_espace(texte_cible)

    seuil_affichage = st.slider(
        "Seuil minimal de probabilité (valence agrégée)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Les phrases dont le score agrégé est inférieur à ce seuil sont masquées dans les résultats.",
        key="camembert_seuil_affichage",
    )
    st.caption(
        "Plus vous augmentez ce seuil, plus seules les phrases dont la valence est clairement"
        " positive ou négative resteront affichées ; un seuil bas laisse passer les phrases"
        " à tonalité plus nuancée."
    )

    if "camembert_pipe" not in st.session_state:
        st.session_state["camembert_pipe"] = None
    if "camembert_resultats" not in st.session_state:
        st.session_state["camembert_resultats"] = None

    if st.button("Lancer l'import CamemBERT", type="primary"):
        with st.spinner("Import et initialisation du modèle CamemBERT..."):
            st.session_state["camembert_pipe"] = _charger_camembert_pipeline()

        if st.session_state["camembert_pipe"] is None:
            st.warning(
                "Le modèle n'a pas pu être chargé. Vérifiez les dépendances puis réessayez."
            )
        else:
            st.success("CamemBERT est prêt pour l'analyse de sentiments.")

    if st.session_state.get("camembert_pipe") is None:
        st.info(
            "Cliquez sur le bouton ci-dessus pour importer et initialiser CamemBERT avant l'analyse."
        )
        return

    resultat_courant = st.session_state.get("camembert_resultats")

    if st.button("Lancer l'analyse CamemBERT"):
        with st.spinner("Inférence en cours..."):
            if st.session_state.get("camembert_pipe") is None:
                st.warning(
                    "Le modèle CamemBERT n'a pas été initialisé. Cliquez d'abord sur le bouton d'import."
                )
                return

            if not texte_cible:
                st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
                return

            phrases = segmenter_en_phrases(texte_cible) or [texte_cible]
            predictions = st.session_state["camembert_pipe"](
                phrases, return_all_scores=True
            )
            df_sentiments = _construire_df_sentiments(phrases, predictions)
            st.session_state["camembert_resultats"] = {
                "df_sentiments": df_sentiments,
                "texte_source": texte_cible,
            }
            resultat_courant = st.session_state["camembert_resultats"]

    if not resultat_courant:
        st.info("Aucune analyse n'a encore été exécutée.")
        return

    df_sentiments = resultat_courant["df_sentiments"]
    df_affiches = df_sentiments[df_sentiments["score_valence"] >= seuil_affichage]

    st.success("Analyse CamemBERT terminée.")
    if df_affiches.empty:
        st.info("Aucun résultat atteint le seuil de probabilité sélectionné.")
        return

    if len(df_affiches) < len(df_sentiments):
        st.caption(
            f"{len(df_affiches)} phrase(s) affichée(s) sur {len(df_sentiments)} après application du seuil."
        )

    st.markdown(_css_camembert_annotations(), unsafe_allow_html=True)
    st.markdown("#### Texte annoté")
    fragment_html = _texte_annote_camembert(df_affiches)
    st.markdown(fragment_html, unsafe_allow_html=True)
    html_complet = (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/><title>Texte annoté CamemBERT</title>"
        f"{_css_camembert_annotations()}</head><body>{fragment_html}</body></html>"
    )
    st.download_button(
        "Télécharger le texte annoté (HTML)",
        data=html_complet.encode("utf-8"),
        file_name="texte_annote_camembert.html",
        mime="text/html",
        key="camembert_dl_html",
    )

    st.markdown("#### Tableau des scores")
    st.dataframe(df_affiches, use_container_width=True)

    st.markdown("#### Graphiques des sentiments")
    _tracer_barres_scores(df_affiches)
    st.markdown("##### Courbe CamemBERT")
    fenetre_lissage = st.slider(
        "Taille de la moyenne glissante pour la courbe CamemBERT (en nombre de phrases)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Augmenter cette valeur permet d'adoucir la courbe de valence affichée ci-dessous.",
        key="camembert_fenetre_lissage",
    )
    _tracer_courbe_valence(df_affiches, fenetre_lissage)
    _tracer_moyennes(df_affiches)
