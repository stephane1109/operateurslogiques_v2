"""Interface Streamlit dédiée à l'analyse IRaMuTeQ.

Cette page d'accueil se concentre uniquement sur les outils disponibles dans le
répertoire ``iramuteq``. Elle permet d'importer un corpus IRaMuTeQ, d'en
segmenter les variables/modalités, puis d'analyser les connecteurs logiques
à partir du dictionnaire ``connecteursiramuteq.json``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from iramuteq.analyseiramuteq import render_corpus_iramuteq_tab
from iramuteq.corpusiramuteq import charger_corpus_iramuteq, lire_fichier_iramuteq

BASE_DIR = Path(__file__).resolve().parent
DICTIONNAIRES_DIR = BASE_DIR / "dictionnaires"
EXEMPLES_DIR = BASE_DIR / "exemples"


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
    if "corpus_hash" not in st.session_state:
        st.session_state.corpus_hash = ""


def vider_cache_application() -> None:
    """Vide le cache Streamlit au démarrage pour garantir un état initial propre."""

    if st.session_state.get("cache_deja_purge"):
        return

    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.cache_deja_purge = True


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


def _charger_et_memoriser_corpus(source, nom_fichier: str) -> None:
    """Lit un corpus, stocke les résultats et son hash dans la session."""

    try:
        texte_corpus = lire_fichier_iramuteq(source)
        df_modalites = charger_corpus_iramuteq(source)
    except Exception as err:
        st.error(f"Impossible de lire le corpus : {err}")
        return

    contenu_bytes = texte_corpus.encode("utf-8", errors="ignore")
    hash_corpus = hashlib.sha256(contenu_bytes).hexdigest()

    if hash_corpus == st.session_state.get("corpus_hash"):
        return

    st.session_state.corpus_df = df_modalites
    st.session_state.corpus_texte = texte_corpus
    st.session_state.corpus_nom = nom_fichier
    st.session_state.corpus_hash = hash_corpus


def _proposer_exemples() -> None:
    """Affiche la liste des fichiers d'exemple locaux et permet de les charger."""

    fichiers_exemple = sorted(EXEMPLES_DIR.glob("*.txt")) if EXEMPLES_DIR.exists() else []
    if not fichiers_exemple:
        st.info("Aucun fichier d'exemple n'a été trouvé dans le dossier 'exemples'.")
        return

    fichier_selectionne = st.selectbox(
        "Ou choisir un discours d'exemple :",
        options=[""] + [f.name for f in fichiers_exemple],
        format_func=lambda x: "Sélectionner..." if x == "" else x,
    )

    charger_exemple = st.button("Charger l'exemple sélectionné", use_container_width=True)

    if charger_exemple and fichier_selectionne:
        chemin_fichier = EXEMPLES_DIR / fichier_selectionne
        _charger_et_memoriser_corpus(chemin_fichier, fichier_selectionne)


def page_iramuteq() -> None:
    """Construit la page principale centrée sur les outils IRaMuTeQ."""

    st.set_page_config(
        page_title="Analyse IRaMuTeQ des connecteurs logiques",
        page_icon="📑",
        layout="wide",
    )

    vider_cache_application()
    initialiser_session()

    with st.sidebar:
        st.header("Importer un corpus")
        st.caption(
            "Chargez un fichier IRaMuTeQ (.txt ou .iramuteq). Le découpage est assuré par le script "
            "`iramuteq/corpusiramuteq.py`."
        )
        fichier_corpus = st.file_uploader(
            "Déposer un corpus IRaMuTeQ (.txt ou .iramuteq)",
            type=["txt", "iramuteq"],
            accept_multiple_files=False,
            help="Le fichier doit contenir les balises **** et les variables/modalités attendues par IRaMuTeQ.",
        )

        if fichier_corpus is not None:
            _charger_et_memoriser_corpus(fichier_corpus, fichier_corpus.name)

        st.divider()
        _proposer_exemples()

    df_modalites = st.session_state.corpus_df

    onglet_import, onglet_analyse = st.tabs(["Importer le corpus", "Analyses"])

    with onglet_import:
        st.title("Importer et préparer le corpus IRaMuTeQ")
        st.markdown(
            "Cette interface utilise les modules **iramuteq** pour importer un corpus, "
            "segmenter les variables/modalités et préparer l'analyse des connecteurs logiques."
        )

        st.markdown(
            """### Comment démarrer ?
            1. Déposez un fichier texte IRaMuTeQ (.txt) ou une archive de projet (.iramuteq).
            2. Vérifiez le découpage automatique du corpus (variable, modalité, texte).
            3. Passez à l'onglet « Analyses » pour explorer les statistiques.
            """
        )

        if df_modalites is not None and not df_modalites.empty:
            st.success(
                f"Corpus chargé : {st.session_state.corpus_nom or 'fichier inconnu'} • {len(st.session_state.corpus_texte)} caractères"
            )
            afficher_resume_corpus(df_modalites)
            with st.expander("Aperçu du corpus segmenté", expanded=False):
                st.dataframe(df_modalites, use_container_width=True)
        else:
            st.info("Aucun corpus chargé pour le moment.")

    with onglet_analyse:
        st.title("Analyses IRaMuTeQ des connecteurs logiques")
        st.markdown(
            "Les statistiques et textes annotés s'appuient sur le dictionnaire « connecteursiramuteq.json »."
        )

        if df_modalites is None or df_modalites.empty:
            st.info(
                "Importez d'abord un corpus via l'onglet « Importer le corpus » pour lancer l'analyse."
            )
            return

        render_corpus_iramuteq_tab(
            df_modalites,
            dictionnaires_dir=DICTIONNAIRES_DIR,
            use_regex_cc=True,
            preparer_detections=None,
        )


def main() -> None:
    """Point d'entrée de l'application Streamlit.

    La page unique se limite à afficher le titre, les menus latéraux d'import
    et les onglets d'analyse dédiés à IRaMuTeQ.
    """

    page_iramuteq()


if __name__ == "__main__":
    main()
