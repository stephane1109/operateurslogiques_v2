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

    st.title("Analyse IRaMuTeQ des connecteurs logiques")
    st.markdown(
        "Cette interface s'appuie exclusivement sur les modules du dossier "
        "**iramuteq** pour importer un corpus, l'explorer par variable/modalité "
        "et visualiser les connecteurs logiques définis dans le dictionnaire "
        "`connecteursiramuteq.json`."
    )

    st.markdown(
        """### Comment démarrer ?
        1. Déposez un fichier texte IRaMuTeQ (.txt) contenant vos balises `****` et vos variables/modalités.
        2. Consultez le découpage automatique du corpus (variable, modalité, texte).
        3. Explorez les statistiques et les textes annotés avec le dictionnaire de connecteurs IRaMuTeQ.
        """
    )

    fichier_corpus = st.file_uploader(
        "Déposer un corpus IRaMuTeQ (.txt)",
        type=["txt"],
        accept_multiple_files=False,
    )

    texte_corpus = ""
    df_modalites = pd.DataFrame()

    if fichier_corpus is not None:
        try:
            texte_corpus, df_modalites = charger_corpus(fichier_corpus)
        except Exception as err:
            st.error(f"Impossible de lire le corpus : {err}")
            return

        if df_modalites.empty:
            st.warning("Aucune variable ou modalité détectée dans le fichier fourni.")
        else:
            st.success(
                f"Corpus chargé : {fichier_corpus.name} • {len(texte_corpus)} caractères"
            )
            afficher_resume_corpus(df_modalites)
            with st.expander("Aperçu du corpus segmenté", expanded=False):
                st.dataframe(df_modalites, use_container_width=True)

    if df_modalites is not None and not df_modalites.empty:
        st.divider()
        render_corpus_iramuteq_tab(
            df_modalites,
            dictionnaires_dir=DICTIONNAIRES_DIR,
            use_regex_cc=True,
            preparer_detections=None,
        )
    else:
        st.info(
            "Importez un corpus IRaMuTeQ pour accéder aux statistiques et aux textes annotés."
        )


if __name__ == "__main__":
    page_iramuteq()
