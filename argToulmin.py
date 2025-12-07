"""Composants d'affichage pour l'onglet Arg Toulmin."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

from toulmin import (
    charger_lexiques_toulmin,
    detecter_toulmin,
    normaliser_texte,
    segmenter_phrases,
)
from streamlit_utils import dataframe_safe


TOULMIN_DEFINITIONS: Dict[str, str] = {
    "CLAIM": (
        "Définition de Claim (thèse) : énoncé principal que l’orateur cherche à faire "
        "accepter. Dans les discours politiques, ce sont les positions, décisions, "
        "promesses, annonces ou jugements centraux."
    ),
    "DATA": (
        "Définition de Data / Grounds (données) : informations factuelles mobilisées "
        "pour soutenir la thèse. Il s’agit de chiffres, exemples, références à des "
        "rapports, citations d’autorité ou descriptions concrètes d’états de faits."
    ),
    "WARRANT": (
        "Définition de Warrant (garant) : principe, règle, présupposé ou schéma "
        "d’inférence qui relie les données à la thèse. Ce sont les justifications "
        "implicites ou explicites du type « donc », « ce qui implique que », « afin "
        "de », « de sorte que », généralisations et raisonnements qui font le pont "
        "entre faits et conclusion."
    ),
    "BACKING": (
        "Définition de Backing (soutien) : appuis qui renforcent le garant, par "
        "exemple textes de loi, institutions, autorités épistémiques, jurisprudence, "
        "directives ou expertise."
    ),
    "QUALIFIER": (
        "Définition de Qualifier (qualificatif) : modalisation du degré de certitude "
        "ou d’extension de la thèse, par exemple « probablement », « en principe », "
        "« dans la plupart des cas », « il est possible que »."
    ),
    "REBUTTAL": (
        "Définition de Rebuttal (réfutation / restriction) : exceptions, "
        "contre-arguments anticipés, conditions limitatives, par exemple « cependant "
        "», « sauf si », « à condition que », « néanmoins »."
    ),
}

# Libellés français → anglais (ordre demandé : traduction FR devant l'indicateur)
TOULMIN_LABELS_FR: Dict[str, str] = {
    "CLAIM": "Thèse (Claim)",
    "DATA": "Données (Data / Grounds)",
    "WARRANT": "Garant (Warrant)",
    "BACKING": "Soutien (Backing)",
    "QUALIFIER": "Modalisation (Qualifier)",
    "REBUTTAL": "Réfutation / Restriction (Rebuttal)",
}

PALETTE_TOULMIN: Dict[str, Dict[str, str]] = {
    "CLAIM": {"bg": "#fff2e6", "fg": "#b35900", "bd": "#b35900"},
    "DATA": {"bg": "#e6f4ff", "fg": "#1565c0", "bd": "#1565c0"},
    "WARRANT": {"bg": "#f3e5f5", "fg": "#7b1fa2", "bd": "#7b1fa2"},
    "BACKING": {"bg": "#e8f5e9", "fg": "#2e7d32", "bd": "#2e7d32"},
    "QUALIFIER": {"bg": "#fffde7", "fg": "#b19b00", "bd": "#b19b00"},
    "REBUTTAL": {"bg": "#fde7f3", "fg": "#ad1457", "bd": "#ad1457"},
}


_ALPHA = "A-Za-zÀ-ÖØ-öø-ÿ"


def _libelle_categorie_toulmin(categorie: str) -> str:
    cat_norm = str(categorie).upper()
    return TOULMIN_LABELS_FR.get(cat_norm, str(categorie).title())


def _esc(texte: str) -> str:
    return html.escape(str(texte))


def _motif_entier(expression: str) -> re.Pattern:
    expr_norm = re.escape(expression.replace("’", "'").replace("`", "'"))
    return re.compile(rf"(?<![{_ALPHA}]){expr_norm}(?![{_ALPHA}])", flags=re.I)


def _compiler_motifs_toulmin(lexiques: Dict[str, Iterable[str]]) -> List[Tuple[str, str, re.Pattern]]:
    motifs: List[Tuple[str, str, re.Pattern]] = []
    for categorie, expressions in lexiques.items():
        cat = categorie.replace("MARQUEURS_", "").upper()
        exprs_tries = sorted(set(expressions), key=lambda s: len(s or ""), reverse=True)
        for expr in exprs_tries:
            if not expr:
                continue
            motifs.append((cat, expr, _motif_entier(normaliser_texte(expr))))
    return motifs


def _occurrences_toulmin(
    texte: str, motifs: List[Tuple[str, str, re.Pattern]]
) -> List[Dict[str, Any]]:
    occs: List[Dict[str, Any]] = []
    for cat, expr, motif in motifs:
        for m in motif.finditer(texte):
            occs.append(
                {
                    "debut": m.start(),
                    "fin": m.end(),
                    "surface": texte[m.start() : m.end()],
                    "categorie": cat,
                    "marqueur": expr,
                    "longueur": m.end() - m.start(),
                }
            )
    occs.sort(key=lambda x: (x["debut"], -x["longueur"]))
    res: List[Dict[str, Any]] = []
    borne = -1
    for m in occs:
        if m["debut"] >= borne:
            res.append(m)
            borne = m["fin"]
    return res


def _css_toulmin_badges() -> str:
    lignes = [
        "<style>",
        "div.texte-annote.toulmin { line-height: 1.6; font-size: 1.05rem; }",
        "div.texte-annote.toulmin span.marque-toulmin { font-weight: 600; text-decoration: underline; }",
        "div.texte-annote.toulmin span.badge-toulmin { display: inline-block; padding: 0.05rem 0.4rem; margin-left: 0.25rem; border: 1px solid #222; border-radius: 0.35rem; font-family: monospace; font-size: 0.82em; vertical-align: baseline; }",
    ]
    for cat, pal in PALETTE_TOULMIN.items():
        lignes.append(
            f"div.texte-annote.toulmin span.badge-{_esc(cat)} {{ background-color: {pal['bg']}; color: {pal['fg']}; border-color: {pal['bd']}; }}"
        )
    lignes.append("</style>")
    return "\n".join(lignes)


def _html_annote_toulmin(
    texte: str,
    lexiques: Dict[str, Iterable[str]],
    show_categories: Dict[str, bool] | None = None,
) -> str:
    if not texte:
        return "<div class='texte-annote toulmin'>(Texte vide)</div>"

    show_categories = {str(cat).upper(): bool(val) for cat, val in (show_categories or {}).items()}

    motifs = _compiler_motifs_toulmin(lexiques)
    occs = _occurrences_toulmin(texte, motifs)

    morceaux: List[str] = []
    curseur = 0
    for occ in occs:
        cat = occ["categorie"].upper()
        if show_categories and not show_categories.get(cat, True):
            continue
        if occ["debut"] > curseur:
            morceaux.append(_esc(texte[curseur : occ["debut"]]))
        css_class = f"badge-toulmin badge-{_esc(cat)}"
        badge = f"<span class='{css_class}'>{_esc(cat.title())}</span>"
        marque = f"<span class='marque-toulmin'>{_esc(occ['surface'])}</span>"
        morceaux.append(marque + badge)
        curseur = occ["fin"]
    if curseur < len(texte):
        morceaux.append(_esc(texte[curseur:]))

    if not morceaux:
        return "<div class='texte-annote toulmin'>(Aucune occurrence dans les catégories sélectionnées)</div>"
    return "<div class='texte-annote toulmin'>" + "".join(morceaux) + "</div>"


def _render_definitions() -> None:
    st.markdown("#### Rappels sur le schéma de Toulmin")
    for cat in ["CLAIM", "DATA", "WARRANT", "BACKING", "QUALIFIER", "REBUTTAL"]:
        definition = TOULMIN_DEFINITIONS.get(cat, "")
        if definition:
            label = _libelle_categorie_toulmin(cat)
            st.markdown(f"- **{label}** — {definition}")


def _preparer_dataframe_toulmin(detections_toulmin: List[Dict[str, Any]]) -> pd.DataFrame:
    df_toulmin = pd.DataFrame(detections_toulmin, columns=["id_phrase", "categorie", "marqueur", "phrase"])
    df_toulmin["categorie_norm"] = df_toulmin["categorie"].str.upper()
    df_toulmin["categorie_label"] = df_toulmin["categorie_norm"].apply(_libelle_categorie_toulmin)
    return df_toulmin


def _render_stats(texte_source: str, df_toulmin: pd.DataFrame) -> None:
    st.markdown("### Statistiques")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Discours**")
        phrases = segmenter_phrases(texte_source)
        nb_mots = len(normaliser_texte(texte_source).split()) if texte_source else 0
        nb_car = len(texte_source)
        nb_phrases = len(phrases)
        st.metric("Mots", nb_mots)
        st.metric("Caractères", nb_car)
        st.metric("Phrases", nb_phrases)
    with col2:
        st.markdown("**Arguments Toulmin**")
        if df_toulmin.empty:
            st.info("Aucune occurrence détectée.")
            return
        total = len(df_toulmin)
        st.metric("Occurrences totales", total)
        repartition = (
            df_toulmin.groupby(["categorie_norm", "categorie_label"])
            .size()
            .reset_index(name="compte")
            .rename(columns={"categorie_label": "Catégorie (FR / EN)"})
        )
        repartition["part_%"] = repartition["compte"].div(total).mul(100).round(1)
        dataframe_safe(
            repartition[["Catégorie (FR / EN)", "compte", "part_%"]],
            use_container_width=True,
            hide_index=True,
        )

        # Graphique de répartition
        chart_data = repartition.rename(columns={"compte": "Occurrences"})
        st.bar_chart(
            chart_data,
            x="Catégorie (FR / EN)",
            y="Occurrences",
        )


def _render_toulmin_analysis(
    texte_source: str,
    lexiques_toulmin: Dict[str, Iterable[str]],
    heading: str,
    key_prefix: str,
    couleur_heading: str | None = None,
) -> None:
    heading_html = html.escape(heading)
    if couleur_heading:
        st.markdown(
            f"<h3 style='margin-bottom:0; color:{couleur_heading};'>{heading_html}</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### {heading_html}")

    if not texte_source.strip():
        st.info("Aucun texte fourni pour ce discours.")
        return

    detections_toulmin = detecter_toulmin(texte_source, lexiques_toulmin)
    df_toulmin = _preparer_dataframe_toulmin(detections_toulmin)

    if df_toulmin.empty:
        st.info("Aucune composante du schéma de Toulmin n'a été identifiée.")
    else:
        df_affichage = df_toulmin.rename(
            columns={
                "id_phrase": "Phrase #",
                "categorie_label": "Catégorie (FR / EN)",
                "marqueur": "Marqueur",
                "phrase": "Phrase",
            }
        )[["Phrase #", "Catégorie (FR / EN)", "Marqueur", "Phrase"]]

        dataframe_safe(df_affichage, use_container_width=True, hide_index=True)
        st.download_button(
            "Exporter les occurrences (CSV)",
            data=df_affichage.to_csv(index=False).encode("utf-8"),
            file_name="toulmin_occurrences.csv",
            mime="text/csv",
            key=f"{key_prefix}dl_toulmin_csv",
        )

    st.markdown("#### Texte annoté (Toulmin)")
    categories = sorted(df_toulmin["categorie_norm"].unique())
    show_categories = {cat: True for cat in categories}
    with st.expander("Afficher / masquer des catégories", expanded=False):
        col_all, col_none = st.columns(2)
        if col_all.button("Tout cocher", key=f"{key_prefix}toulmin_all"):
            for cat in categories:
                st.session_state[f"{key_prefix}chk_toulmin_{cat}"] = True
        if col_none.button("Tout décocher", key=f"{key_prefix}toulmin_none"):
            for cat in categories:
                st.session_state[f"{key_prefix}chk_toulmin_{cat}"] = False
        for cat in categories:
            label = _libelle_categorie_toulmin(cat)
            show_categories[cat] = st.checkbox(
                label,
                value=st.session_state.get(f"{key_prefix}chk_toulmin_{cat}", True),
                key=f"{key_prefix}chk_toulmin_{cat}",
            )

    st.markdown(_css_toulmin_badges(), unsafe_allow_html=True)
    frag = _html_annote_toulmin(
        texte_source,
        lexiques_toulmin,
        show_categories=show_categories,
    )
    st.markdown(frag, unsafe_allow_html=True)
    st.download_button(
        "Exporter le texte annoté (HTML)",
        data=frag.encode("utf-8"),
        file_name="texte_toulmin_annote.html",
        mime="text/html",
        key=f"{key_prefix}dl_toulmin_html",
    )

    _render_stats(texte_source, df_toulmin)


def render_toulmin_tab(
    texte_source: str,
    texte_source_2: str | None = None,
    heading_discours_1: str = "Discours 1",
    heading_discours_2: str = "Discours 2",
    couleur_discours_1: str = "#c00000",
    couleur_discours_2: str = "#1f4e79",
) -> None:
    """Affiche tout le contenu de l'onglet Arg Toulmin."""

    st.subheader("Analyse argumentative — schéma de Toulmin")
    st.caption(
        "Repérage des composantes CLAIM / DATA / WARRANT / BACKING / QUALIFIER / REBUTTAL à partir d'expressions clés."
    )

    _render_definitions()

    if not texte_source.strip() and not (texte_source_2 and texte_source_2.strip()):
        st.info("Aucun texte fourni.")
        return

    lexiques_toulmin = charger_lexiques_toulmin()

    with st.expander("Lexiques utilisés (argumToulmin.json)", expanded=False):
        st.json(lexiques_toulmin, expanded=False)

    has_disc1 = bool(texte_source and texte_source.strip())
    has_disc2 = bool(texte_source_2 and texte_source_2.strip())

    if has_disc1 and has_disc2:
        st.markdown("#### Analyses des deux discours")
        col1, col2 = st.columns(2)
        with col1:
            _render_toulmin_analysis(
                texte_source,
                lexiques_toulmin,
                heading_discours_1,
                key_prefix="disc1_",
                couleur_heading=couleur_discours_1,
            )
        with col2:
            _render_toulmin_analysis(
                texte_source_2,
                lexiques_toulmin,
                heading_discours_2,
                key_prefix="disc2_",
                couleur_heading=couleur_discours_2,
            )
    elif has_disc1:
        _render_toulmin_analysis(
            texte_source,
            lexiques_toulmin,
            heading_discours_1,
            key_prefix="disc1_",
            couleur_heading=couleur_discours_1,
        )
    else:
        _render_toulmin_analysis(
            texte_source_2 or "",
            lexiques_toulmin,
            heading_discours_2,
            key_prefix="disc2_",
            couleur_heading=couleur_discours_2,
        )
