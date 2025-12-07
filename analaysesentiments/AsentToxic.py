"""Onglet d'analyse de toxicité basé sur CamemBERT pour l'application Streamlit."""
from __future__ import annotations

from typing import Dict, List

import html

import pandas as pd
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, CamembertTokenizer, pipeline

from text_utils import normaliser_espace, segmenter_en_phrases


ETIQUETTES_CANONIQUES = {
    "toxic": "Toxic",
    "toxique": "Toxic",
    "toxicite": "Toxic",
    "non-toxic": "Non-Toxic",
    "nontoxic": "Non-Toxic",
    "non_toxic": "Non-Toxic",
    "non toxique": "Non-Toxic",
    "sensible": "Sensible",
    "sensitivity": "Sensible",
}
COULEURS_TOXICITE = {
    "Toxic": "indianred",
    "Non-Toxic": "seagreen",
    "Sensible": "goldenrod",
}
BADGES_TOXICITE = {"Toxic": "⚠️", "Non-Toxic": "✅", "Sensible": "🚸"}


@st.cache_resource(show_spinner=False)
def _charger_toxicite_pipeline():
    """Charge la pipeline CamemBERT spécialisée pour la toxicité."""

    try:
        try:
            tokenizer = CamembertTokenizer.from_pretrained(
                "AgentPublic/camembert-base-toxic-fr-user-prompts"
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                "AgentPublic/camembert-base-toxic-fr-user-prompts", use_fast=False
            )
        return pipeline(
            "text-classification",
            model="AgentPublic/camembert-base-toxic-fr-user-prompts",
            tokenizer=tokenizer,
        )
    except Exception as exc:  # pragma: no cover - uniquement déclenché en environnement Streamlit
        st.error(
            "Impossible de charger le modèle de toxicité. Vérifiez la connexion et les dépendances nécessaires."
        )
        st.exception(exc)
        return None


def _normaliser_etiquette(etiquette: str) -> str:
    etiquette_norm = etiquette.strip().lower().replace("_", "-")
    etiquette_norm = "-".join(etiquette_norm.split())
    return ETIQUETTES_CANONIQUES.get(etiquette_norm, etiquette.strip())


def _construire_df_toxicite(phrases: List[str], predictions) -> pd.DataFrame:
    """Transforme les scores du modèle en DataFrame lisible par phrase."""

    if not predictions:
        return pd.DataFrame(
            columns=["id_phrase", "texte_phrase", "toxicite", "score_toxicite"]
        )

    lignes = []
    for idx, (phrase, scores) in enumerate(zip(phrases, predictions), start=1):
        if not scores:
            continue

        meilleur = max(scores, key=lambda sc: sc.get("score", 0))
        etiquette = _normaliser_etiquette(str(meilleur.get("label", "")))
        ligne = {
            "id_phrase": idx,
            "texte_phrase": phrase,
            "toxicite": etiquette,
            "score_toxicite": float(meilleur.get("score", 0)),
        }

        for score in scores:
            etiquette_score = _normaliser_etiquette(str(score.get("label", "")))
            etiquette_slug = (
                etiquette_score.lower().replace(" ", "_").replace("-", "_")
            )
            ligne[f"score_{etiquette_slug}"] = float(score.get("score", 0))

        lignes.append(ligne)

    return pd.DataFrame(lignes)


def _css_toxicite_annotations() -> str:
    return """
<style>
    .tox-texte-annote {
        line-height: 1.7;
        font-size: 1.02rem;
        white-space: pre-wrap;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .tox-annotation {
        display: inline-flex;
        align-items: flex-start;
        padding: 0.5rem 0.65rem;
        margin: 0.1rem 0;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        color: #111;
    }
    .tox-phrase { margin-left: 0.4rem; display: inline-block; color: #111; }
    .tox-badge { font-weight: 700; font-size: 0.92rem; }
    .tox-Toxic { background-color: #fde8e7; border-color: #f09a96; }
    .tox-Non-Toxic { background-color: #e9f7ef; border-color: #7cc78d; }
    .tox-Sensible { background-color: #fff8e6; border-color: #e0c175; }
</style>
"""


def _texte_annote_toxicite(df_toxicite: pd.DataFrame) -> str:
    if df_toxicite.empty:
        return "<div class='tox-texte-annote'>(Aucune phrase à afficher)</div>"

    lignes = []
    for ligne in df_toxicite.itertuples():
        etiquette = str(ligne.toxicite)
        badge = BADGES_TOXICITE.get(etiquette, "🔎")
        classe = f"tox-{etiquette}"
        texte_phrase = html.escape(str(ligne.texte_phrase))
        lignes.append(
            (
                f"<div class='tox-annotation {classe}'>"
                f"<span class='tox-badge'>{badge} {html.escape(etiquette)}</span>"
                f"<span class='tox-phrase'>{texte_phrase}</span>"
                "</div>"
            )
        )

    return "<div class='tox-texte-annote'>" + "".join(lignes) + "</div>"


def _tracer_barres(df_toxicite: pd.DataFrame):
    if df_toxicite.empty:
        st.info("Aucune phrase à représenter.")
        return

    chart = (
        alt.Chart(df_toxicite)
        .mark_bar()
        .encode(
            x=alt.X("id_phrase:O", title="Phrase"),
            y=alt.Y("score_toxicite:Q", title="Score (classe dominante)"),
            color=alt.Color(
                "toxicite:N",
                title="Classe",
                scale=alt.Scale(
                    domain=list(COULEURS_TOXICITE.keys()),
                    range=list(COULEURS_TOXICITE.values()),
                ),
            ),
            tooltip=[
                "id_phrase",
                "toxicite",
                alt.Tooltip("score_toxicite:Q", format=".3f"),
            ],
        )
        .properties(height=300, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _tracer_moyennes(df_toxicite: pd.DataFrame):
    colonnes_scores = [col for col in df_toxicite.columns if col.startswith("score_")]
    if not colonnes_scores:
        return

    df_moyennes = (
        df_toxicite[colonnes_scores]
        .mean()
        .reset_index()
        .rename(columns={"index": "classe", 0: "score"})
    )
    df_moyennes["classe"] = (
        df_moyennes["classe"]
        .str.replace("score_", "", regex=False)
        .str.replace("_", " ")
        .apply(_normaliser_etiquette)
    )

    chart = (
        alt.Chart(df_moyennes)
        .mark_bar()
        .encode(
            x=alt.X("classe:N", title="Classe"),
            y=alt.Y("score:Q", title="Score moyen"),
            color=alt.Color(
                "classe:N",
                scale=alt.Scale(
                    domain=list(COULEURS_TOXICITE.keys()),
                    range=list(COULEURS_TOXICITE.values()),
                ),
            ),
            tooltip=["classe", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=250, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _selectionner_texte(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
) -> str:
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
            help="Le texte sélectionné est pré-rempli ci-dessous pour l'inférence Toxicité.",
            key="toxicite_discours_select",
        )

    contenu_initial = textes_disponibles.get(choix, "") if choix else ""
    return st.text_area(
        "Texte à analyser",
        value=contenu_initial
        or "C'est formidable de vous voir, même si certains propos ont été très virulents.",
        height=200,
        key="toxicite_text_area",
    )


def render_toxicite_tab(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
):
    """Rendu Streamlit pour l'onglet de toxicité CamemBERT."""

    st.markdown("### AnalysSentToxicCamemBERT")
    st.caption(
        "Modèle CamemBERT spécialisé pour la détection de toxicité et de contenu sensible."
    )

    st.markdown(
        """
        **Modèle utilisé :** [AgentPublic/camembert-base-toxic-fr-user-prompts](https://huggingface.co/AgentPublic/camembert-base-toxic-fr-user-prompts)

        **Étiquettes :** `Toxic`, `Non-Toxic`, `Sensible`.
        Le modèle est entraîné sur des prompts utilisateurs en français pour distinguer
        la toxicité, l'absence de toxicité et le contenu sensible.
        """
    )

    st.markdown(
        """
        **Comment fonctionne cette analyse ?**

        * Le texte est découpé en phrases pour évaluer finement la toxicité de chaque segment.
        * Le modèle retourne la probabilité de chaque classe (Toxic / Non-Toxic / Sensible).
        * Les tableaux et graphiques ci-dessous permettent d'explorer la répartition
          de ces scores.
        """
    )

    texte_cible = _selectionner_texte(texte_discours_1, texte_discours_2, nom_discours_1, nom_discours_2)
    texte_cible = normaliser_espace(texte_cible)

    seuil_affichage = st.slider(
        "Seuil minimal de probabilité (classe dominante)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Les phrases dont le score de classe dominante est inférieur à ce seuil sont masquées dans les résultats.",
        key="toxicite_seuil_affichage",
    )

    if "toxicite_pipe" not in st.session_state:
        st.session_state["toxicite_pipe"] = None
    if "toxicite_resultats" not in st.session_state:
        st.session_state["toxicite_resultats"] = None

    if st.button("Charger le modèle Toxicité", type="primary"):
        with st.spinner("Import et initialisation du modèle de toxicité..."):
            st.session_state["toxicite_pipe"] = _charger_toxicite_pipeline()

        if st.session_state["toxicite_pipe"] is None:
            st.warning(
                "Le modèle n'a pas pu être chargé. Vérifiez les dépendances puis réessayez."
            )
        else:
            st.success("Modèle Toxicité prêt pour l'analyse.")

    if st.session_state.get("toxicite_pipe") is None:
        st.info(
            "Cliquez sur le bouton ci-dessus pour importer et initialiser le modèle de toxicité avant l'analyse."
        )
        return

    resultat_courant = st.session_state.get("toxicite_resultats")

    if st.button("Lancer l'analyse Toxicité"):
        with st.spinner("Inférence en cours..."):
            if st.session_state.get("toxicite_pipe") is None:
                st.warning(
                    "Le modèle de toxicité n'a pas été initialisé. Cliquez d'abord sur le bouton d'import."
                )
                return

            if not texte_cible:
                st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
                return

            phrases = segmenter_en_phrases(texte_cible) or [texte_cible]
            predictions = st.session_state["toxicite_pipe"](
                phrases, return_all_scores=True
            )
            df_toxicite = _construire_df_toxicite(phrases, predictions)
            st.session_state["toxicite_resultats"] = {
                "df_toxicite": df_toxicite,
                "texte_source": texte_cible,
            }
            resultat_courant = st.session_state["toxicite_resultats"]

    if not resultat_courant:
        st.info("Aucune analyse n'a encore été exécutée.")
        return

    df_toxicite = resultat_courant["df_toxicite"]
    df_affiches = df_toxicite[df_toxicite["score_toxicite"] >= seuil_affichage]

    st.success("Analyse de toxicité terminée.")
    if df_affiches.empty:
        st.info("Aucun résultat atteint le seuil de probabilité sélectionné.")
        return

    if len(df_affiches) < len(df_toxicite):
        st.caption(
            f"{len(df_affiches)} phrase(s) affichée(s) sur {len(df_toxicite)} après application du seuil."
        )

    st.markdown(_css_toxicite_annotations(), unsafe_allow_html=True)
    st.markdown("#### Texte annoté")
    fragment_html = _texte_annote_toxicite(df_affiches)
    st.markdown(fragment_html, unsafe_allow_html=True)
    html_complet = (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/><title>Texte annoté Toxicité</title>"
        f"{_css_toxicite_annotations()}</head><body>{fragment_html}</body></html>"
    )
    st.download_button(
        "Télécharger le texte annoté (HTML)",
        data=html_complet.encode("utf-8"),
        file_name="texte_annote_toxicite.html",
        mime="text/html",
        key="toxicite_dl_html",
    )

    st.markdown("#### Tableau des scores")
    st.dataframe(df_affiches, use_container_width=True)

    st.markdown("#### Graphiques")
    _tracer_barres(df_affiches)
    _tracer_moyennes(df_affiches)
