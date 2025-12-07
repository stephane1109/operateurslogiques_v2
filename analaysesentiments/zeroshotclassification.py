"""Onglet d'analyse Zero-Shot Classification"""
from __future__ import annotations

from typing import List

import streamlit as st
from transformers import pipeline

from text_utils import normaliser_espace


@st.cache_resource(show_spinner=False)
def _charger_zero_shot_pipeline():
    """Charge la pipeline Zero-Shot multilingue."""

    try:
        return pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        )
    except Exception as exc:  # pragma: no cover - uniquement déclenché en environnement Streamlit
        st.error(
            "Impossible de charger le modèle Zero-Shot. Vérifiez la connexion et les dépendances nécessaires."
        )
        st.exception(exc)
        return None


def _selectionner_texte(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
) -> str:
    """Permet de choisir rapidement entre les deux discours ou saisir un texte libre."""

    textes_disponibles = {}
    if texte_discours_1.strip():
        textes_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        textes_disponibles[nom_discours_2] = texte_discours_2

    choix = None
    if textes_disponibles:
        choix = st.selectbox(
            "Choisissez un discours à charger dans la zone de test",
            options=list(textes_disponibles.keys()),
            help="Le texte sélectionné est pré-rempli ci-dessous pour l'inférence Zero-Shot.",
            key="zero_shot_discours_select",
        )

    contenu_initial = textes_disponibles.get(choix, "") if choix else ""
    return st.text_area(
        "Texte à analyser",
        value=contenu_initial
        or "Ces derniers temps, je pense de plus en plus à en finir.",
        height=200,
        key="zero_shot_text_area",
    )


def _afficher_resultats(labels: List[str], scores: List[float]):
    """Affiche les scores sous forme de tableau trié."""

    if not labels or not scores:
        st.info("Aucun résultat à afficher.")
        return

    st.markdown("#### Résultats (probabilités par étiquette)")
    st.dataframe(
        {
            "étiquette": labels,
            "probabilité": [round(score, 3) for score in scores],
        },
        use_container_width=True,
    )


def render_zero_shot_tab(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
):
    """Rendu Streamlit pour l'onglet Zero-Shot Classification."""

    st.markdown("### zeroclassification")
    st.caption(
        "Analyse Zero-Shot multilingue basée sur MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

    st.markdown(
        """
        **Modèle utilisé :** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` pour la tâche
        *zero-shot-classification*.

        Le modèle « Zero-Shot multilingue » est conçu pour fonctionner sur plusieurs
        langues, dont le français (le dataset XNLI utilisé pour ce modèle couvre 15
        langues, y compris le français).

        Référence Hugging Face :
        https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
        """
    )

    st.markdown(
        """
        Exemple d'usage :

        ```python
        phrase_suicide = "Ces derniers temps, je pense de plus en plus à en finir."
        labels = ["espoir", "détresse", "problème", "solution", "neutre"]
        ```

        Définissez ci-dessous vos propres étiquettes (séparées par des virgules) pour obtenir la probabilité
        d'appartenance du texte à chaque catégorie.
        """
    )

    texte_cible = _selectionner_texte(texte_discours_1, texte_discours_2, nom_discours_1, nom_discours_2)
    texte_cible = normaliser_espace(texte_cible)

    etiquettes_str = st.text_input(
        "Étiquettes (séparées par des virgules)",
        value="espoir, détresse, problème, solution, neutre",
        help="Renseignez les labels à tester pour la classification Zero-Shot.",
    )
    etiquettes = [label.strip() for label in etiquettes_str.split(",") if label.strip()]

    if "zero_shot_pipe" not in st.session_state:
        st.session_state["zero_shot_pipe"] = None

    if st.button("Charger le modèle Zero-Shot", type="primary"):
        with st.spinner("Import et initialisation du modèle Zero-Shot..."):
            st.session_state["zero_shot_pipe"] = _charger_zero_shot_pipeline()

        if st.session_state["zero_shot_pipe"] is None:
            st.warning(
                "Le modèle n'a pas pu être chargé. Vérifiez les dépendances puis réessayez."
            )
        else:
            st.success("Modèle Zero-Shot prêt pour l'analyse.")

    if st.session_state.get("zero_shot_pipe") is None:
        st.info(
            "Cliquez sur le bouton ci-dessus pour importer et initialiser le modèle Zero-Shot avant l'analyse."
        )
        return

    if st.button("Lancer l'analyse Zero-Shot"):
        if not texte_cible:
            st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
            return
        if not etiquettes:
            st.warning("Veuillez définir au moins une étiquette.")
            return

        with st.spinner("Inférence en cours..."):
            resultat = st.session_state["zero_shot_pipe"](
                sequences=texte_cible,
                candidate_labels=etiquettes,
                multi_label=True,
            )

        if not resultat:
            st.error("Aucun résultat retourné par le modèle.")
            return

        _afficher_resultats(resultat.get("labels", []), resultat.get("scores", []))
