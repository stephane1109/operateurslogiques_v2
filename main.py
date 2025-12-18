"""Interface Streamlit dédiée à l'analyse IRaMuTeQ.

Cette page d'accueil se concentre uniquement sur les outils disponibles dans le
répertoire ``iramuteq``. Elle permet d'importer un corpus IRaMuTeQ, d'en
segmenter les variables/modalités, puis d'analyser les connecteurs logiques
à partir du dictionnaire ``connecteursiramuteq.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from iramuteq.analyseiramuteq import render_corpus_iramuteq_tab
from iramuteq.corpusiramuteq import segmenter_corpus_par_modalite

BASE_DIR = Path(__file__).resolve().parent
DICTIONNAIRES_DIR = BASE_DIR / "dictionnaires"


def lire_fichier_txt(uploaded_file) -> str:
    """Lit un fichier texte en essayant plusieurs encodages courants."""

    if uploaded_file is None:
        return ""

    donnees = uploaded_file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return donnees.decode(enc)
        except Exception:
            continue
    return donnees.decode("utf-8", errors="ignore")


def initialiser_session() -> None:
    """Prépare les clés de session nécessaires pour partager le corpus entre les pages."""

    if "corpus_df" not in st.session_state:
        st.session_state.corpus_df = pd.DataFrame(
            columns=["variable", "modalite", "texte", "balise"]
        )
    if "corpus_texte" not in st.session_state:
        st.session_state.corpus_texte = ""
    if "corpus_nom" not in st.session_state:
        st.session_state.corpus_nom = ""


def vider_cache_application() -> None:
    """Vide le cache Streamlit au démarrage pour garantir un état initial propre."""

    if st.session_state.get("cache_deja_purge"):
        return

    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.cache_deja_purge = True


def charger_corpus(uploaded_file) -> Tuple[str, pd.DataFrame]:
    """Retourne le texte du corpus et son découpage en variables/modalités."""

    texte = lire_fichier_txt(uploaded_file)
    if not texte.strip():
        return "", pd.DataFrame(columns=["variable", "modalite", "texte", "balise"])

    df_modalites = segmenter_corpus_par_modalite(texte)
    return texte, df_modalites


def afficher_resume_corpus(df_modalites: pd.DataFrame) -> None:
    """Affiche un résumé synthétique du corpus importé."""

    variables = sorted({v for v in df_modalites.get("variable", []).dropna() if str(v).strip()})
    modalites = sorted({m for m in df_modalites.get("modalite", []).dropna() if str(m).strip()})

    col1, col2, col3 = st.columns(3)
    col1.metric("Segments détectés", f"{len(df_modalites):,}".replace(",", " "))
    col2.metric("Variables", f"{len(variables):,}".replace(",", " "))
    col3.metric("Modalités", f"{len(modalites):,}".replace(",", " "))

    if variables:
        st.caption(
            "Variables trouvées : " + ", ".join(variables)
        )


def page_iramuteq() -> None:
    """Construit la page principale centrée sur les outils IRaMuTeQ."""

    st.set_page_config(
        page_title="Analyse IRaMuTeQ des connecteurs logiques",
        page_icon="📑",
        layout="wide",
    )

    vider_cache_application()
    initialiser_session()

    st.sidebar.header("Navigation")
    page_courante = st.sidebar.radio(
        "Aller à",
        (
            "Importer le corpus",
            "Analyser les connecteurs",
        ),
    )

    fichier_corpus = st.sidebar.file_uploader(
        "Déposer un corpus IRaMuTeQ (.txt)",
        type=["txt"],
        accept_multiple_files=False,
        help="Le fichier doit contenir les balises **** et les variables/modalités attendues par IRaMuTeQ.",
    )

    if fichier_corpus is not None:
        try:
            texte_corpus, df_modalites = charger_corpus(fichier_corpus)
        except Exception as err:
            st.error(f"Impossible de lire le corpus : {err}")
            return

        st.session_state.corpus_df = df_modalites
        st.session_state.corpus_texte = texte_corpus
        st.session_state.corpus_nom = fichier_corpus.name

    df_modalites = st.session_state.corpus_df

    if page_courante == "Importer le corpus":
        st.title("Importer et préparer le corpus IRaMuTeQ")
        st.markdown(
            "Cette interface utilise les modules **iramuteq** pour importer un corpus, "
            "segmenter les variables/modalités et préparer l'analyse des connecteurs logiques."
        )

        st.markdown(
            """### Comment démarrer ?
            1. Déposez un fichier texte IRaMuTeQ (.txt) contenant vos balises `****` et vos variables/modalités.
            2. Vérifiez le découpage automatique du corpus (variable, modalité, texte).
            3. Passez à la page « Analyser les connecteurs » pour explorer les statistiques.
            """
        )

        if fichier_corpus is None and df_modalites.empty:
            st.info("Aucun corpus chargé pour le moment.")

        if df_modalites is not None and not df_modalites.empty:
            st.success(
                f"Corpus chargé : {st.session_state.corpus_nom or 'fichier inconnu'} • {len(st.session_state.corpus_texte)} caractères"
            )
            afficher_resume_corpus(df_modalites)
            with st.expander("Aperçu du corpus segmenté", expanded=False):
                st.dataframe(df_modalites, use_container_width=True)

    if page_courante == "Analyser les connecteurs":
        st.title("Analyse IRaMuTeQ des connecteurs logiques")
        st.markdown(
            "Les statistiques et textes annotés s'appuient sur le dictionnaire « connecteursiramuteq.json »."
        )

        if df_modalites is None or df_modalites.empty:
            st.info(
                "Importez d'abord un corpus via la page « Importer le corpus » pour lancer l'analyse."
            )
            return

        render_corpus_iramuteq_tab(
            df_modalites,
            dictionnaires_dir=DICTIONNAIRES_DIR,
            use_regex_cc=True,
            preparer_detections=None,
        )


if __name__ == "__main__":
    page_iramuteq()
