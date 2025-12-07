"""Outils de statistiques et de visualisation pour les marqueurs détectés."""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd
import streamlit as st
import altair as alt


def _segmenter_en_phrases_local(texte: str) -> List[str]:
    """Segmente approximativement le texte en phrases en s'alignant sur la logique de main.py."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]


def _positions_phrases(texte: str):
    """Calcule pour chaque phrase l’offset caractère de début pour projeter les positions en % du texte."""
    phrases = _segmenter_en_phrases_local(texte)
    offsets = []
    pos = 0
    t = texte
    for ph in phrases:
        i = t.find(ph, pos)
        if i < 0:
            i = pos
        offsets.append(i)
        pos = i + len(ph)
    return phrases, offsets, len(texte)


def _ajoute_colonne_t_rel(
    df: pd.DataFrame,
    phrases: List[str],
    offsets: List[int],
    total_len: int,
    col_id: str = "id_phrase",
    col_pos: str = "position",
):
    """Ajoute une colonne t_rel ∈ [0,100], position relative du marqueur dans le texte."""
    if df.empty:
        return df
    m = df.copy()

    def t_rel_from_row(r: pd.Series) -> float:
        try:
            i = int(r[col_id]) - 1
            base = offsets[i] if 0 <= i < len(offsets) else 0
            pos_abs = base + int(r.get(col_pos, 0))
        except Exception:
            pos_abs = 0
        if total_len <= 0:
            return 0.0
        return round(100.0 * pos_abs / total_len, 3)

    m["t_rel"] = m.apply(t_rel_from_row, axis=1)
    return m


def construire_df_temps(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marq: pd.DataFrame,
    df_memoires: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
    df_tensions: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne tous les jeux en un seul tableau temporel avec colonnes normalisées pour Altair."""
    if not texte_source or not texte_source.strip():
        return pd.DataFrame()

    phrases, offsets, total_len = _positions_phrases(texte_source)

    df_conn = df_conn if df_conn is not None else pd.DataFrame()
    df_marq = df_marq if df_marq is not None else pd.DataFrame()
    df_memoires = df_memoires if df_memoires is not None else pd.DataFrame()
    df_consq_lex = df_consq_lex if df_consq_lex is not None else pd.DataFrame()
    df_causes_lex = df_causes_lex if df_causes_lex is not None else pd.DataFrame()
    df_tensions = df_tensions if df_tensions is not None else pd.DataFrame()

    a = df_conn.copy()
    if not a.empty:
        a["type"] = "CONNECTEUR"
        a.rename(columns={"connecteur": "surface", "code": "etiquette"}, inplace=True)
        a = _ajoute_colonne_t_rel(a, phrases, offsets, total_len)

    b = df_marq.copy()
    if not b.empty:
        b["type"] = "MARQUEUR"
        b.rename(columns={"marqueur": "surface", "categorie": "etiquette"}, inplace=True)
        b = _ajoute_colonne_t_rel(b, phrases, offsets, total_len)

    m = df_memoires.copy()
    if not m.empty:
        m["type"] = "MEMOIRE"
        m.rename(columns={"memoire": "surface", "categorie": "etiquette"}, inplace=True)
        m = _ajoute_colonne_t_rel(m, phrases, offsets, total_len)

    c = df_consq_lex.copy()
    if not c.empty:
        c["type"] = "CONSEQUENCE"
        c.rename(columns={"consequence": "surface", "categorie": "etiquette"}, inplace=True)
        c = _ajoute_colonne_t_rel(c, phrases, offsets, total_len)

    d = df_causes_lex.copy()
    if not d.empty:
        d["type"] = "CAUSE"
        d.rename(columns={"cause": "surface", "categorie": "etiquette"}, inplace=True)
        d = _ajoute_colonne_t_rel(d, phrases, offsets, total_len)
    t = df_tensions.copy()
    if not t.empty:
        t["type"] = "TENSION SÉMANTIQUE"
        t.rename(columns={"expression": "surface", "tension": "etiquette"}, inplace=True)
        t = _ajoute_colonne_t_rel(t, phrases, offsets, total_len)

    frames = [x for x in [a, b, m, c, d, t] if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["etiquette"] = (
        df["etiquette"].astype(str).str.strip().replace("", pd.NA).fillna("INCONNU")
    )
    cols = [
        "t_rel",
        "id_phrase",
        "surface",
        "etiquette",
        "type",
        "position",
        "longueur",
        "phrase",
    ]

    valeurs_par_defaut = {
        "t_rel": 0.0,
        "id_phrase": 0,
        "surface": "",
        "etiquette": "",
        "type": "",
        "position": 0,
        "longueur": 0,
        "phrase": "",
    }

    for col in cols:
        if col not in df.columns:
            df[col] = valeurs_par_defaut[col]

    # Homogénéise les types pour éviter des erreurs Altair/Streamlit
    df["t_rel"] = pd.to_numeric(df["t_rel"], errors="coerce").fillna(0.0)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype(int)
    df["longueur"] = pd.to_numeric(df["longueur"], errors="coerce").fillna(0).astype(int)
    df["id_phrase"] = (
        pd.to_numeric(df["id_phrase"], errors="coerce").fillna(0).astype(int)
    )
    df["surface"] = df["surface"].astype(str)
    df["etiquette"] = df["etiquette"].astype(str)
    df["type"] = df["type"].astype(str)
    df["phrase"] = df["phrase"].astype(str)

    return df[cols]


def graphique_altair_chronologie(
    df_temps: pd.DataFrame,
    filtres_types: List[str] | None = None,
    filtres_etiquettes: List[str] | None = None,
):
    """Construit un scatter Altair chronologique; filtres optionnels sur type et étiquette."""
    if df_temps.empty:
        return None

    data = df_temps.copy()
    if filtres_types:
        data = data[data["type"].isin(filtres_types)]
    if filtres_etiquettes:
        data = data[data["etiquette"].isin([e.upper() for e in filtres_etiquettes])]

    base = (
        alt.Chart(data)
        .mark_point(filled=True)
        .encode(
            x=alt.X(
                "t_rel:Q",
                title="Progression du discours (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("etiquette:N", title="marqueurs"),
            color=alt.Color("type:N", title="Type", legend=alt.Legend(title="Type")),
            size=alt.Size("longueur:Q", title="Longueur repérée", legend=None),
            tooltip=[
                alt.Tooltip("t_rel:Q", title="Progression (%)", format=".2f"),
                alt.Tooltip("id_phrase:Q", title="Phrase #"),
                alt.Tooltip("surface:N", title="mot"),
                alt.Tooltip("etiquette:N", title="Marqueur"),
                alt.Tooltip("type:N", title="Type"),
            ],
        )
        .properties(height=320)
    )

    return base


def graphique_barres_marqueurs_temps(
    df_temps_marqueurs: pd.DataFrame,
):
    """Construit un graphique des occurrences par typologie de marqueur."""

    if df_temps_marqueurs is None or df_temps_marqueurs.empty:
        return None

    colonnes_requises = {"etiquette"}
    if not colonnes_requises.issubset(df_temps_marqueurs.columns):
        return None

    df_freq = (
        df_temps_marqueurs
        .assign(etiquette=lambda d: d["etiquette"].astype(str))
        .groupby(["etiquette"], dropna=False)
        .size()
        .reset_index(name="occurrences")
    )

    if df_freq.empty:
        return None

    df_freq.sort_values(by=["occurrences", "etiquette"], ascending=[False, True], inplace=True)

    chart = (
        alt.Chart(df_freq)
        .mark_bar()
        .encode(
            x=alt.X("occurrences:Q", title="Occurrences dans le discours"),
            y=alt.Y("etiquette:N", title="marqueurs", sort="-x"),
            color=alt.Color("etiquette:N", title="Typologie"),
            tooltip=[
                alt.Tooltip("etiquette:N", title="Marqueur"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .properties(height=320)
    )
    return chart


def graphique_barres_familles_connecteurs(
    df_conn: pd.DataFrame,
):
    """Construit un graphique de la répartition des Connecteurs logiques par famille."""

    if df_conn is None or df_conn.empty:
        return None

    if "code" not in df_conn.columns:
        return None

    series_codes = df_conn["code"].dropna().astype(str).str.strip()
    series_codes = series_codes[series_codes != ""]
    if series_codes.empty:
        return None

    df_freq = (
        series_codes.str.upper()
        .value_counts()
        .rename_axis("code")
        .reset_index(name="occurrences")
    )

    if df_freq.empty:
        return None

    df_freq.sort_values(by=["occurrences", "code"], ascending=[False, True], inplace=True)

    chart = (
        alt.Chart(df_freq)
        .mark_bar()
        .encode(
            x=alt.X("occurrences:Q", title="Occurrences dans le discours"),
            y=alt.Y("code:N", title="Famille", sort="-x"),
            color=alt.Color("code:N", title="Famille"),
            tooltip=[
                alt.Tooltip("code:N", title="Famille"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .properties(height=320)
    )
    return chart


def _normaliser_couleur(choix: Optional[str], valeur_par_defaut: str) -> str:
    """Restreint les couleurs aux variantes rouge ou bleu pour les titres."""

    if not choix:
        return valeur_par_defaut

    choix_min = choix.strip().lower()
    couleurs_autorisees = {
        "rouge": "#c00000",
        "red": "#c00000",
        "bleu": "#1f4e79",
        "blue": "#1f4e79",
        "#c00000": "#c00000",
        "#1f4e79": "#1f4e79",
    }

    return couleurs_autorisees.get(choix_min, valeur_par_defaut)


def _render_stats_block(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marqueurs: pd.DataFrame,
    df_memoires: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
    df_tensions: pd.DataFrame,
    heading: Optional[str] = None,
    heading_color: Optional[str] = None,
    key_prefix: str = "",
) -> None:
    """Affiche les statistiques des marqueurs pour un discours donné."""
    if heading:
        couleur_titre = _normaliser_couleur(heading_color, heading_color or "#c00000")
        st.markdown(
            f'<span style="color:{couleur_titre}; font-weight:700; font-size:26px;">{heading}</span>',
            unsafe_allow_html=True,
        )
    st.subheader("Familles de Connecteurs logiques")
    chart_familles_conn = graphique_barres_familles_connecteurs(df_conn)
    if chart_familles_conn is None:
        st.info("Aucun Connecteur logique détecté pour générer la répartition par famille.")
    else:
        st.altair_chart(chart_familles_conn, use_container_width=True)

    st.markdown("---")

    st.subheader("Statistiques des marqueurs normatifs, mémoire & tensions sémantiques")

    df_temps = construire_df_temps(
        texte_source=texte_source,
        df_conn=df_conn,
        df_marq=df_marqueurs,
        df_memoires=df_memoires,
        df_consq_lex=df_consq_lex,
        df_causes_lex=df_causes_lex,
        df_tensions=df_tensions,
    )

    df_normatif = pd.DataFrame()
    if df_marqueurs is not None and not df_marqueurs.empty:
        df_normatif = df_marqueurs.copy()
        df_normatif["categorie"] = df_normatif["categorie"].astype(str).str.upper()
        df_normatif["surface"] = df_normatif["marqueur"].astype(str)
        df_normatif["type_source"] = "NORMATIF"

    df_memoire = pd.DataFrame()
    if df_memoires is not None and not df_memoires.empty:
        df_memoire = df_memoires.copy()
        df_memoire["categorie"] = df_memoire["categorie"].astype(str).str.upper()
        df_memoire["surface"] = df_memoire["memoire"].astype(str)
        df_memoire["type_source"] = "MEMOIRE"

    df_tension = pd.DataFrame()
    if df_tensions is not None and not df_tensions.empty:
        df_tension = df_tensions.copy()
        df_tension["categorie"] = df_tension["tension"].astype(str).str.upper()
        df_tension["surface"] = df_tension["expression"].astype(str)
        df_tension["type_source"] = "TENSION SÉMANTIQUE"

    frames = [x for x in [df_normatif, df_memoire, df_tension] if not x.empty]

    if not frames:
        st.info("Aucun marqueur détecté pour générer des statistiques.")
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, ignore_index=True)

        total_occurrences = len(df)
        total_categories = df["categorie"].nunique(dropna=True)
        total_surfaces_distinctes = df["surface"].nunique(dropna=True)
        total_phrases = (
            df["id_phrase"].nunique(dropna=True) if "id_phrase" in df.columns else None
        )

        cols = st.columns(4 if total_phrases is not None else 3)
        cols[0].metric("Occurrences (total)", f"{total_occurrences}")
        cols[1].metric("Catégories distinctes", f"{total_categories}")
        cols[2].metric("Expressions distinctes", f"{total_surfaces_distinctes}")
        if total_phrases is not None:
            cols[3].metric("Phrases concernées", f"{total_phrases}")

        repartition_parts = []
        if not df_normatif.empty:
            repartition_parts.append(f"normatifs : {len(df_normatif)}")
        if not df_memoire.empty:
            repartition_parts.append(f"mémoire : {len(df_memoire)}")
        if not df_tension.empty:
            repartition_parts.append(f"tensions : {len(df_tension)}")
        if repartition_parts:
            st.caption("Répartition — " + " ; ".join(repartition_parts))

        st.markdown("### Fréquence des marqueurs")
        df_temps_marqueurs = (
            df_temps[
                df_temps["type"].isin(
                    ["MARQUEUR", "MEMOIRE", "TENSION SÉMANTIQUE"]
                )
            ].copy()
            if df_temps is not None and not df_temps.empty
            else pd.DataFrame()
        )
        if df_temps_marqueurs.empty:
            st.info("Impossible de construire la distribution temporelle des marqueurs.")
        else:
            chart_marqueurs_temps = graphique_barres_marqueurs_temps(df_temps_marqueurs)
            if chart_marqueurs_temps is None:
                st.info("Rien à afficher avec les données disponibles.")
            else:
                st.altair_chart(chart_marqueurs_temps, use_container_width=True)

        st.markdown("---")

    if df.empty:
        st.markdown("---")

    st.markdown("### Chronologie des marqueurs")
    if df_temps.empty:
        options_familles: List[str] = []
    else:
        etiquettes = (
            df_temps["etiquette"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
        )
        options_familles = sorted(str(e) for e in etiquettes)

    choix_familles = st.multiselect(
        "Filtrer par famille/catégorie",
        options_familles,
        default=[],
        key=f"{key_prefix}select_familles_chrono",
    )

    if df_temps.empty:
        st.info("Aucune détection pour construire la chronologie.")
    else:
        chart = graphique_altair_chronologie(
            df_temps,
            filtres_etiquettes=choix_familles,
        )
        if chart is None:
            st.info("Rien à afficher avec les filtres actuels.")
        else:
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Chaque point représente une occurrence détectée, positionnée en pourcentage du texte."
            )


def render_stats_tab(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marqueurs: pd.DataFrame,
    df_memoires: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
    df_tensions: pd.DataFrame,
    texte_source_2: Optional[str] = None,
    df_conn_2: Optional[pd.DataFrame] = None,
    df_marqueurs_2: Optional[pd.DataFrame] = None,
    df_memoires_2: Optional[pd.DataFrame] = None,
    df_consq_lex_2: Optional[pd.DataFrame] = None,
    df_causes_lex_2: Optional[pd.DataFrame] = None,
    df_tensions_2: Optional[pd.DataFrame] = None,
    heading_discours_1: Optional[str] = None,
    heading_discours_2: Optional[str] = None,
    couleur_discours_1: Optional[str] = None,
    couleur_discours_2: Optional[str] = None,
) -> None:
    """Affiche les statistiques des marqueurs dans l'onglet Streamlit dédié.

    Lorsque deux discours sont fournis, les indicateurs sont présentés successivement
    (discours 1 puis discours 2) avec les titres colorés.
    """

    _render_stats_block(
        texte_source,
        df_conn,
        df_marqueurs,
        df_memoires,
        df_consq_lex,
        df_causes_lex,
        df_tensions,
        heading=heading_discours_1,
        heading_color=couleur_discours_1,
        key_prefix="disc1_",
    )

    if texte_source_2 and texte_source_2.strip():
        st.markdown("---")
        _render_stats_block(
            texte_source_2,
            df_conn_2 if df_conn_2 is not None else pd.DataFrame(),
            df_marqueurs_2 if df_marqueurs_2 is not None else pd.DataFrame(),
            df_memoires_2 if df_memoires_2 is not None else pd.DataFrame(),
            df_consq_lex_2 if df_consq_lex_2 is not None else pd.DataFrame(),
            df_causes_lex_2 if df_causes_lex_2 is not None else pd.DataFrame(),
            df_tensions_2 if df_tensions_2 is not None else pd.DataFrame(),
            heading=heading_discours_2,
            heading_color=couleur_discours_2,
            key_prefix="disc2_",
        )

