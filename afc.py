# -*- coding: utf-8 -*-
"""Onglet AFC pour l'analyse de discours.

Ce module propose une préparation des données par phrase, la construction
d'une matrice "phrases × mots" et l'application d'une analyse factorielle
des correspondances (AFC) avec des variables illustratives correspondant
aux marqueurs et connecteurs détectés.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import unicodedata

import prince
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOP_WORDS

from text_utils import normaliser_espace, segmenter_en_phrases


# ===============================================================
# Fonctions utilitaires (préparation des données)
# ===============================================================
def _normaliser_nom_variable(prefixe: str, categorie: str) -> str:
    """Normalise un libellé de catégorie pour en faire un nom de colonne.

    - Passage en minuscules
    - Suppression des accents
    - Remplacement des caractères non alphanumériques par des underscores
    """

    base = unicodedata.normalize("NFD", str(categorie)).encode("ascii", "ignore").decode("utf-8")
    base = base.lower()
    nettoye = "".join(ch if ch.isalnum() else "_" for ch in base)
    nettoye = "_".join(filter(None, nettoye.split("_")))
    return f"{prefixe}{nettoye}" if prefixe else nettoye


def _ajouter_colonnes_booleennes(
    df_phrases: pd.DataFrame,
    df_detection: pd.DataFrame,
    colonne_categorie: str,
    prefixe: str,
) -> None:
    """Crée des colonnes booléennes pour chaque catégorie détectée.

    Les colonnes sont initialisées à False puis positionnées à True pour
    les phrases concernées.
    """

    if df_detection.empty or colonne_categorie not in df_detection.columns:
        return

    df_detection = df_detection.copy()
    df_detection["id_phrase"] = (
        pd.to_numeric(df_detection.get("id_phrase", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    categories = sorted({cat for cat in df_detection[colonne_categorie].dropna().unique() if str(cat).strip()})
    for cat in categories:
        nom_colonne = _normaliser_nom_variable(prefixe, cat)
        if nom_colonne not in df_phrases.columns:
            df_phrases[nom_colonne] = False
        ids = df_detection.loc[df_detection[colonne_categorie] == cat, "id_phrase"].tolist()
        if ids:
            df_phrases.loc[df_phrases["id_phrase"].isin(ids), nom_colonne] = True


def construire_df_phrases(
    texte: str,
    detections: Dict[str, pd.DataFrame],
    libelle_discours: str,
) -> pd.DataFrame:
    """Construit un DataFrame par phrase avec colonnes booléennes pour marqueurs/connecteurs."""

    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm) if texte_norm else []
    df_phrases = pd.DataFrame(
        {
            "id_phrase": list(range(1, len(phrases) + 1)),
            "texte_phrase": phrases,
            "discours": libelle_discours,
        }
    )

    # Ajout des colonnes booléennes par catégorie de détection
    _ajouter_colonnes_booleennes(df_phrases, detections.get("df_marq", pd.DataFrame()), "categorie", "marqueur_")
    _ajouter_colonnes_booleennes(df_phrases, detections.get("df_conn", pd.DataFrame()), "code", "connecteur_")
    _ajouter_colonnes_booleennes(df_phrases, detections.get("df_memoires", pd.DataFrame()), "categorie", "memoire_")
    _ajouter_colonnes_booleennes(df_phrases, detections.get("df_causes_lex", pd.DataFrame()), "categorie", "cause_")
    _ajouter_colonnes_booleennes(df_phrases, detections.get("df_consq_lex", pd.DataFrame()), "categorie", "consequence_")
    _ajouter_colonnes_booleennes(df_phrases, detections.get("df_tensions", pd.DataFrame()), "tension", "tension_")

    return df_phrases


def preparer_matrice_afc(
    df_phrases: pd.DataFrame,
    colonnes_marqueurs: Sequence[str],
    min_df: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, CountVectorizer]:
    """Filtre les phrases pertinentes et construit la matrice phrases × mots."""

    if df_phrases.empty:
        raise ValueError("Aucune phrase disponible pour l'AFC.")

    colonnes_marqueurs = list(colonnes_marqueurs)
    if not colonnes_marqueurs:
        raise ValueError("Aucun marqueur ou connecteur sélectionné.")

    manquantes = [col for col in colonnes_marqueurs if col not in df_phrases.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {', '.join(manquantes)}")

    masque_selection = df_phrases[colonnes_marqueurs].any(axis=1)
    df_selection = df_phrases.loc[masque_selection].copy()
    if df_selection.empty:
        raise ValueError("Aucune phrase ne contient les marqueurs sélectionnés.")

    labels = [f"{row.discours} – phrase {row.id_phrase}" for row in df_selection.itertuples()]
    vectorizer = CountVectorizer(stop_words=sorted(SPACY_STOP_WORDS), min_df=min_df)
    matrice_sparse = vectorizer.fit_transform(df_selection["texte_phrase"])
    if matrice_sparse.shape[1] == 0:
        raise ValueError("Aucun mot retenu après filtrage (stopwords ou fréquence minimale trop élevée).")

    matrice_mots = pd.DataFrame(
        matrice_sparse.toarray(),
        index=labels,
        columns=vectorizer.get_feature_names_out(),
    )
    return df_selection, matrice_mots, vectorizer


def lancer_afc(matrice_mots: pd.DataFrame, n_composantes: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, prince.CA]:
    """Applique l'AFC via prince et renomme les dimensions."""

    ca = prince.CA(n_components=max(2, n_composantes), n_iter=10, copy=True, random_state=0)
    ca = ca.fit(matrice_mots)
    row_df = ca.row_coordinates(matrice_mots)
    col_df = ca.column_coordinates(matrice_mots)

    row_df.columns = [f"Dim {i+1}" for i in range(row_df.shape[1])]
    col_df.columns = [f"Dim {i+1}" for i in range(col_df.shape[1])]
    return row_df, col_df, ca


def calculer_barycentres_marqueurs(
    row_coords: pd.DataFrame,
    df_phrases: pd.DataFrame,
    colonnes_marqueurs: Sequence[str],
    axe_x: int,
    axe_y: int,
) -> pd.DataFrame:
    """Calcule les positions barycentriques des colonnes booléennes sélectionnées."""

    colonnes = [col for col in colonnes_marqueurs if col in df_phrases.columns]
    barycentres = []
    for col in colonnes:
        ids = df_phrases.loc[df_phrases[col], "label_afc"]
        if ids.empty:
            continue
        coords_col = row_coords.loc[ids, [f"Dim {axe_x}", f"Dim {axe_y}"]].mean()
        barycentres.append({
            "libelle": col,
            "Dim 1": coords_col.iloc[0],
            "Dim 2": coords_col.iloc[1],
            "Type": "Marqueur/Connecteur",
        })
    return pd.DataFrame(barycentres)


def _extraire_mots_contributifs(
    col_coords: pd.DataFrame,
    axe_x: int,
    axe_y: int,
    nb_mots: int,
) -> pd.DataFrame:
    """Sélectionne les mots les plus éloignés de l'origine sur les axes demandés."""

    if col_coords.empty:
        return col_coords
    dims = [f"Dim {axe_x}", f"Dim {axe_y}"]
    scores = (col_coords[dims] ** 2).sum(axis=1)
    principaux = scores.nlargest(nb_mots).index
    return col_coords.loc[principaux, dims].reset_index().rename(columns={"index": "libelle"}).assign(Type="Mot")


def tracer_nuage_factoriel(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    df_phrases: pd.DataFrame,
    coords_marqueurs: pd.DataFrame,
    axe_x: int,
    axe_y: int,
    nb_mots: int,
) -> alt.Chart:
    """Construit le nuage factoriel (axes sélectionnés)."""

    dims = [f"Dim {axe_x}", f"Dim {axe_y}"]

    data_phrases = (
        row_coords[dims]
        .reset_index()
        .rename(columns={"index": "libelle"})
        .merge(df_phrases[["label_afc", "texte_phrase", "discours"]], left_on="libelle", right_on="label_afc", how="left")
        .assign(Type="Phrase")
    )

    data_mots = _extraire_mots_contributifs(col_coords, axe_x, axe_y, nb_mots)
    data_marqueurs = coords_marqueurs[["libelle", "Dim 1", "Dim 2", "Type"]] if not coords_marqueurs.empty else pd.DataFrame(columns=["libelle", "Dim 1", "Dim 2", "Type"])

    couches = []
    couleur = alt.condition(alt.datum.Type == "Phrase", alt.value("#1f77b4"), alt.value("#d62728"))

    couches.append(
        alt.Chart(data_phrases)
        .mark_circle(size=80, opacity=0.75)
        .encode(
            x=dims[0],
            y=dims[1],
            color=couleur,
            tooltip=["libelle", "discours", "texte_phrase", alt.Tooltip(dims[0], format=".3f"), alt.Tooltip(dims[1], format=".3f")],
        )
    )

    if not data_mots.empty:
        couches.append(
            alt.Chart(data_mots)
            .mark_text(color="#444", fontSize=12, dy=-6)
            .encode(x="Dim 1", y="Dim 2", text="libelle", tooltip=["libelle", alt.Tooltip("Dim 1", format=".3f"), alt.Tooltip("Dim 2", format=".3f")])
        )

    if not data_marqueurs.empty:
        couches.append(
            alt.Chart(data_marqueurs)
            .mark_point(shape="triangle", size=180, color="#f28e2c")
            .encode(x="Dim 1", y="Dim 2", tooltip=["libelle", alt.Tooltip("Dim 1", format=".3f"), alt.Tooltip("Dim 2", format=".3f")])
        )

    return alt.layer(*couches).properties(height=480, width="container").resolve_scale(color="independent", shape="independent")


# ===============================================================
# Rendu Streamlit
# ===============================================================
def render_afc_tab(
    texte_source: str,
    texte_source_2: str,
    detections_1: Dict[str, pd.DataFrame],
    detections_2: Dict[str, pd.DataFrame],
    libelle_discours_1: str,
    libelle_discours_2: str,
) -> None:
    """Rendu de l'onglet AFC (phrases × mots, variables illustratives marqueurs)."""

    st.subheader("Analyse factorielle des correspondances (AFC)")
    st.caption(
        "Projection des phrases et du lexique, avec les marqueurs/connecteurs comme variables illustratives."
    )

    df_phrases_1 = construire_df_phrases(texte_source, detections_1, libelle_discours_1)
    df_phrases_2 = construire_df_phrases(texte_source_2, detections_2, libelle_discours_2)

    bool_cols = set(df_phrases_1.select_dtypes(include="bool").columns).union(
        df_phrases_2.select_dtypes(include="bool").columns
    )
    for df in (df_phrases_1, df_phrases_2):
        for col in bool_cols:
            if col not in df.columns:
                df[col] = False
            else:
                df[col] = df[col].fillna(False).astype(bool)

    df_phrases = pd.concat([df_phrases_1, df_phrases_2], ignore_index=True)

    if df_phrases.empty:
        st.info("Aucune phrase disponible pour lancer l'AFC.")
        return

    colonnes_marqueurs = [col for col in df_phrases.columns if df_phrases[col].dtype == bool]
    if not colonnes_marqueurs:
        st.info("Aucun marqueur ou connecteur détecté pour construire l'AFC.")
        return

    st.markdown("### Paramètres")
    col_sel, col_freq, col_axes = st.columns([2, 1, 1])
    selection_cols = col_sel.multiselect(
        "Colonnes de marqueurs/connecteurs à utiliser",
        options=sorted(colonnes_marqueurs),
        default=sorted(colonnes_marqueurs),
    )
    min_df = col_freq.slider("Fréquence minimale des mots", min_value=1, max_value=5, value=1, help="Nombre minimal de phrases dans lesquelles un mot doit apparaître pour être conservé.")
    axe_x = int(col_axes.number_input("Axe X", min_value=1, max_value=4, value=1, step=1))
    axe_y = int(col_axes.number_input("Axe Y", min_value=1, max_value=4, value=2, step=1))
    nb_mots = col_axes.slider("Mots affichés", min_value=5, max_value=50, value=15, step=5)

    if axe_x == axe_y:
        st.warning("Veuillez sélectionner deux axes différents pour l'affichage.")
        return

    try:
        df_selection, matrice_mots, vectorizer = preparer_matrice_afc(df_phrases, selection_cols, min_df)
    except Exception as exc:  # pragma: no cover - interaction Streamlit
        st.error(f"Préparation impossible : {exc}")
        return

    df_selection = df_selection.copy()
    df_selection["label_afc"] = [f"{row.discours} – phrase {row.id_phrase}" for row in df_selection.itertuples()]

    try:
        row_df, col_df, ca = lancer_afc(matrice_mots, n_composantes=max(axe_x, axe_y))
    except Exception as exc:  # pragma: no cover - interaction Streamlit
        st.error(f"AFC impossible : {exc}")
        return

    if max(axe_x, axe_y) > row_df.shape[1]:
        st.warning("Le nombre d'axes demandé dépasse les dimensions disponibles pour l'AFC.")
        return

    coords_marqueurs = calculer_barycentres_marqueurs(row_df, df_selection, selection_cols, axe_x, axe_y)
    chart = tracer_nuage_factoriel(row_df, col_df, df_selection, coords_marqueurs, axe_x, axe_y, nb_mots)

    inertie = ca.eigenvalues_.sum()
    explained_inertia = getattr(ca, "explained_inertia_", [])
    if (explained_inertia is None or len(explained_inertia) == 0) and inertie:
        explained_inertia = ca.eigenvalues_ / inertie  # Compatibilité pour les versions de prince sans explained_inertia_

    inertie_dim1 = explained_inertia[axe_x - 1] * 100 if len(explained_inertia) >= axe_x else 0.0
    inertie_dim2 = explained_inertia[axe_y - 1] * 100 if len(explained_inertia) >= axe_y else 0.0

    st.markdown(
        f"**Inertie totale** : {inertie:.3f} · Dim {axe_x} : {inertie_dim1:.1f}% · Dim {axe_y} : {inertie_dim2:.1f}%"
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Coordonnées factorielles (phrases)"):
        st.dataframe(row_df[[f"Dim {axe_x}", f"Dim {axe_y}"]], use_container_width=True)

    with st.expander("Coordonnées factorielles (mots)"):
        st.dataframe(col_df[[f"Dim {axe_x}", f"Dim {axe_y}"]], use_container_width=True)

    if not coords_marqueurs.empty:
        with st.expander("Positions des marqueurs/connecteurs (illustratifs)"):
            st.dataframe(coords_marqueurs, use_container_width=True)

