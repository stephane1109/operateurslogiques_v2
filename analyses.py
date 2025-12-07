"""Composants dédiés à l'onglet "Analyses"."""

from typing import Any, Dict, List, Optional, Tuple
import html
import re

import pandas as pd
import streamlit as st

CODE_VERS_PYTHON: Dict[str, str] = {
    "CONDITION": "CONDITION (SI)",
    "ALORS": "ALORS",
    "ALTERNATIVE": "ALTERNATIVE (Sinon)",
    "WHILE": "while",
    # Compatibilité ascendante
    "IF": "CONDITION (SI)",
    "ELSE": "ALTERNATIVE (Sinon)",
    "AND": "and",
    "OR": "or",
}

COULEURS_BADGES: Dict[str, Dict[str, str]] = {
    "CONDITION": {"bg": "#ffeaea", "fg": "#c00000", "bd": "#c00000"},
    "ALORS": {"bg": "#ffeaea", "fg": "#c00000", "bd": "#c00000"},
    "ALTERNATIVE": {"bg": "#ffeaea", "fg": "#c00000", "bd": "#c00000"},
    "WHILE": {"bg": "#e9fbe6", "fg": "#2f7d32", "bd": "#2f7d32"},
    # Compatibilité ascendante
    "IF": {"bg": "#ffeaea", "fg": "#c00000", "bd": "#c00000"},
    "ELSE": {"bg": "#ffeaea", "fg": "#c00000", "bd": "#c00000"},
    "AND": {"bg": "#e6fffb", "fg": "#0d9488", "bd": "#0d9488"},
    "OR": {"bg": "#fff3e6", "fg": "#b54b00", "bd": "#b54b00"},
}

COULEURS_MARQUEURS: Dict[str, Dict[str, str]] = {
    "OBLIGATION": {"bg": "#fff7e6", "fg": "#a86600", "bd": "#a86600"},
    "INTERDICTION": {"bg": "#ffe6e6", "fg": "#c62828", "bd": "#c62828"},
    "PERMISSION": {"bg": "#e6fff3", "fg": "#2e7d32", "bd": "#2e7d32"},
    "RECOMMANDATION": {"bg": "#eef2ff", "fg": "#3f51b5", "bd": "#3f51b5"},
    "SANCTION": {"bg": "#fde7f3", "fg": "#ad1457", "bd": "#ad1457"},
    "CADRE_OUVERTURE": {"bg": "#e6f7ff", "fg": "#0277bd", "bd": "#0277bd"},
    "CADRE_FERMETURE": {"bg": "#ede7f6", "fg": "#6a1b9a", "bd": "#6a1b9a"},
    "CONSEQUENCE": {"bg": "#fff0f0", "fg": "#b00020", "bd": "#b00020"},
    "CAUSE": {"bg": "#f0fff4", "fg": "#2f855a", "bd": "#2f855a"},
    "MEM_PERS": {"bg": "#e8f4ff", "fg": "#1565c0", "bd": "#1565c0"},
    "MEM_COLL": {"bg": "#f1f8e9", "fg": "#33691e", "bd": "#33691e"},
    "MEM_RAPPEL": {"bg": "#fff3e0", "fg": "#ef6c00", "bd": "#ef6c00"},
    "MEM_RENVOI": {"bg": "#f3e5f5", "fg": "#7b1fa2", "bd": "#7b1fa2"},
    "MEM_REPET": {"bg": "#ede7f6", "fg": "#5e35b1", "bd": "#5e35b1"},
    "MEM_PASSE": {"bg": "#efebe9", "fg": "#6d4c41", "bd": "#6d4c41"},
}

COULEURS_TENSIONS: Dict[str, Dict[str, str]] = {
    "DEFAULT": {"bg": "#fef3ff", "fg": "#7b1fa2", "bd": "#7b1fa2"},
}


def _esc(s: str) -> str:
    return html.escape(s, quote=False)


def construire_regex_depuis_liste(expressions: List[str]) -> List[Tuple[str, re.Pattern]]:
    """Construit des motifs regex en priorisant les locutions longues (gestion d'apostrophes)."""
    exprs_tries = sorted(expressions, key=lambda s: len(s), reverse=True)
    motifs = []
    for e in exprs_tries:
        e_norm = re.escape(e.replace("’", "'"))
        motif = re.compile(
            rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){e_norm}(?![A-Za-zÀ-ÖØ-öø-ÿ])", flags=re.I
        )
        motifs.append((e, motif))
    return motifs


def libelle_python(code: str) -> str:
    return CODE_VERS_PYTHON.get(str(code).upper(), str(code).upper())


def css_badges() -> str:
    lignes = [
        "<style>",
        ".texte-annote { line-height: 1.8; font-size: 1.05rem; white-space: pre-wrap; }",
        ".badge-code { display: inline-block; padding: 0.05rem 0.4rem; margin-left: 0.25rem; border: 1px solid #333; border-radius: 0.35rem; font-family: monospace; font-size: 0.85em; vertical-align: baseline; }",
        ".badge-marqueur { display: inline-block; padding: 0.03rem 0.35rem; margin-left: 0.2rem; border: 1px dashed #333; border-radius: 0.35rem; font-family: monospace; font-size: 0.78em; vertical-align: baseline; }",
        ".badge-tension { display: inline-block; padding: 0.03rem 0.35rem; margin-left: 0.2rem; border: 1px dashed #333; border-radius: 0.35rem; font-family: monospace; font-size: 0.78em; vertical-align: baseline; background-color: #fef3ff; color: #7b1fa2; border-color: #7b1fa2; }",
        ".connecteur { font-weight: 600; color: #c00000; }",
        ".mot-marque { font-weight: 600; text-decoration: underline; }",
        ".mot-tension { font-weight: 600; text-decoration: underline dotted; }",
        "</style>",
    ]
    for code, pal in COULEURS_BADGES.items():
        lignes.insert(
            -1,
            f".badge-code.code-{code} {{ background-color: {pal['bg']}; color: {pal['fg']}; border-color: {pal['bd']}; }}",
        )
    for cat, pal in COULEURS_MARQUEURS.items():
        lignes.insert(
            -1,
            f".badge-marqueur.marq-{_esc(cat)} {{ background-color: {pal['bg']}; color: {pal['fg']}; border-color: {pal['bd']}; }}",
        )
    for cat, pal in COULEURS_TENSIONS.items():
        lignes.insert(
            -1,
            f".badge-tension.tension-{_esc(cat)} {{ background-color: {pal['bg']}; color: {pal['fg']}; border-color: {pal['bd']}; }}",
        )
    return "\n".join(lignes)


def occurrences_mixte(
    texte: str,
    dico_conn: Dict[str, str],
    dico_marq: Dict[str, str],
    dico_memoires: Dict[str, str],
    dico_consq: Dict[str, str],
    dico_causes: Dict[str, str],
    dico_tensions: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Fusionne toutes les occurrences, élimine les chevauchements (priorité au plus long)."""
    occs: List[Dict[str, Any]] = []
    for typ, dico in [
        ("connecteur", dico_conn),
        ("marqueur", dico_marq),
        ("memoire", dico_memoires),
        ("consequence", dico_consq),
        ("cause", dico_causes),
        ("tension", dico_tensions),
    ]:
        if not dico:
            continue
        motifs = construire_regex_depuis_liste(list(dico.keys()))
        for cle, motif in motifs:
            for m in motif.finditer(texte):
                occs.append(
                    {
                        "debut": m.start(),
                        "fin": m.end(),
                        "original": texte[m.start() : m.end()],
                        "type": typ,
                        "cle": cle,
                        "etiquette": dico[cle],
                        "longueur": m.end() - m.start(),
                    }
                )
    occs.sort(key=lambda x: (x["debut"], -x["longueur"]))
    res = []
    borne = -1
    for m in occs:
        if m["debut"] >= borne:
            res.append(m)
            borne = m["fin"]
    return res


def html_annote(
    texte: str,
    dico_conn: Dict[str, str],
    dico_marq: Dict[str, str],
    dico_memoires: Dict[str, str],
    dico_consq: Dict[str, str],
    dico_causes: Dict[str, str],
    dico_tensions: Dict[str, str],
    show_codes: Dict[str, bool],
    show_consequences: bool,
    show_causes: bool,
    show_tensions: bool,
    show_marqueurs_categories: Optional[Dict[str, bool]] = None,
    show_memoires_categories: Optional[Dict[str, bool]] = None,
    show_tensions_categories: Optional[Dict[str, bool]] = None,
) -> str:
    """Produit le HTML annoté selon les cases cochées."""
    if not texte:
        return "<div class='texte-annote'>(Texte vide)</div>"
    if show_marqueurs_categories is not None:
        show_marqueurs_categories = {
            str(cat).upper(): bool(val) for cat, val in show_marqueurs_categories.items()
        }
    if show_memoires_categories is not None:
        show_memoires_categories = {
            str(cat).upper(): bool(val) for cat, val in show_memoires_categories.items()
        }
    if show_tensions_categories is not None:
        show_tensions_categories = {
            str(cat).upper(): bool(val) for cat, val in show_tensions_categories.items()
        }

    t = texte
    occ = occurrences_mixte(
        t, dico_conn, dico_marq, dico_memoires, dico_consq, dico_causes, dico_tensions
    )
    morceaux: List[str] = []
    curseur = 0
    for m in occ:
        # Filtres d’affichage
        if m["type"] == "connecteur":
            code = str(m["etiquette"]).upper()
            if not show_codes.get(code, True):
                continue
        elif m["type"] == "marqueur":
            cat = str(m["etiquette"]).upper()
            if (
                show_marqueurs_categories is not None
                and not show_marqueurs_categories.get(cat, True)
            ):
                continue
        elif m["type"] == "memoire":
            cat = str(m["etiquette"]).upper()
            if show_memoires_categories is not None and not show_memoires_categories.get(
                cat, True
            ):
                continue
        elif m["type"] == "consequence":
            if not show_consequences:
                continue
        elif m["type"] == "cause":
            if not show_causes:
                continue
        elif m["type"] == "tension":
            if not show_tensions:
                continue
            cat = str(m["etiquette"]).upper()
            if show_tensions_categories is not None and not show_tensions_categories.get(
                cat, True
            ):
                continue

        if m["debut"] > curseur:
            morceaux.append(_esc(t[curseur : m["debut"]]))

        mot_original = _esc(t[m["debut"] : m["fin"]])
        if m["type"] == "connecteur":
            code = str(m["etiquette"]).upper()
            badge = (
                f"<span class='badge-code code-{_esc(code)}'>{_esc(libelle_python(code))}</span>"
            )
            rendu = f"<span class='connecteur'>{mot_original}</span>{badge}"
        elif m["type"] == "tension":
            cat_disp = str(m["etiquette"]).upper()
            css_suffix = cat_disp if cat_disp in COULEURS_TENSIONS else "DEFAULT"
            badge = (
                f"<span class='badge-tension tension-{_esc(css_suffix)}'>{_esc(cat_disp)}</span>"
            )
            rendu = f"<span class='mot-tension'>{mot_original}</span>{badge}"
        else:
            cat_disp = str(m["etiquette"]).upper()
            badge = (
                f"<span class='badge-marqueur marq-{_esc(cat_disp)}'>{_esc(cat_disp)}</span>"
            )
            rendu = f"<span class='mot-marque'>{mot_original}</span>{badge}"

        morceaux.append(rendu)
        curseur = m["fin"]

    if curseur < len(t):
        morceaux.append(_esc(t[curseur:]))

    if not morceaux:
        return "<div class='texte-annote'>(Aucune annotation selon les cases sélectionnées)</div>"
    return "<div class='texte-annote'>" + "".join(morceaux) + "</div>"


def toggle_checkboxes(prefix: str, options_keys: List[str], value: bool) -> None:
    """Force un ensemble de cases à cocher Streamlit à True/False via st.session_state."""
    for opt in options_keys:
        key = f"{prefix}{opt}"
        st.session_state[key] = value


def html_autonome(fragment_html: str) -> str:
    return (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/><title>Texte annoté</title>"
        f"{css_badges()}</head><body>{fragment_html}</body></html>"
    )


def render_detection_section(
    texte_source: str,
    detections: Dict[str, pd.DataFrame],
    *,
    key_prefix: str = "",
    use_regex_cc: bool = True,
    heading_color: Optional[str] = None,
    dico_connecteurs: Dict[str, str],
    dico_marqueurs: Dict[str, str],
    dico_memoires: Dict[str, str],
    dico_consq: Dict[str, str],
    dico_causes: Dict[str, str],
    dico_tensions: Dict[str, str],
) -> None:
    """Affiche les résultats de détection pour un texte (onglet Analyses)."""

    df_conn = detections.get("df_conn", pd.DataFrame())
    df_marq = detections.get("df_marq", pd.DataFrame())
    df_memoires = detections.get("df_memoires", pd.DataFrame())
    df_consq_lex = detections.get("df_consq_lex", pd.DataFrame())
    df_causes_lex = detections.get("df_causes_lex", pd.DataFrame())
    df_tensions = detections.get("df_tensions", pd.DataFrame())

    def _subheader(titre: str) -> None:
        if heading_color:
            st.markdown(
                f"<span style=\"color:{heading_color}; font-size:1.25rem; font-weight:700;\">{html.escape(titre)}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.subheader(titre)

    _subheader("Connecteurs logiques détectés")
    if df_conn.empty:
        st.info("Aucun Connecteur logique détecté ou aucun texte fourni.")
    else:
        st.dataframe(df_conn)
        st.download_button(
            "Exporter Connecteurs logiques (CSV)",
            data=df_conn.to_csv(index=False).encode("utf-8"),
            file_name="occurrences_connecteurs_logiques.csv",
            mime="text/csv",
            key=f"{key_prefix}dl_occ_conn_csv",
        )

    _subheader("Marqueurs détectés")
    if df_marq.empty:
        st.info("Aucun marqueur détecté.")
    else:
        st.dataframe(df_marq)
        st.download_button(
            "Exporter marqueurs (CSV)",
            data=df_marq.to_csv(index=False).encode("utf-8"),
            file_name="occurrences_marqueurs.csv",
            mime="text/csv",
            key=f"{key_prefix}dl_occ_marq_csv",
        )

    _subheader("Tensions sémantiques détectées")
    if not dico_tensions:
        st.info("Aucun dictionnaire de tensions sémantiques chargé.")
    elif df_tensions.empty:
        st.info("Aucune tension sémantique détectée dans le texte.")
    else:
        st.dataframe(df_tensions)
        st.download_button(
            "Exporter tensions sémantiques (CSV)",
            data=df_tensions.to_csv(index=False).encode("utf-8"),
            file_name="tensions_semantiques.csv",
            mime="text/csv",
            key=f"{key_prefix}dl_occ_tensions_csv",
        )

    _subheader("Marqueurs mémoire détectés")
    if df_memoires.empty:
        st.info("Aucun marqueur mémoire détecté.")
    else:
        st.dataframe(df_memoires)
        st.download_button(
            "Exporter marqueurs mémoire (CSV)",
            data=df_memoires.to_csv(index=False).encode("utf-8"),
            file_name="occurrences_memoires.csv",
            mime="text/csv",
            key=f"{key_prefix}dl_occ_memoires_csv",
        )

    colX, colY = st.columns(2)
    with colX:
        _subheader("Déclencheurs de conséquence (Regex)")
        if not use_regex_cc:
            st.info("Méthode Regex désactivée (voir barre latérale).")
        elif df_consq_lex.empty:
            st.info("Aucun déclencheur de conséquence détecté par Regex.")
        else:
            st.dataframe(df_consq_lex)
            st.download_button(
                "Exporter conséquences (CSV)",
                data=df_consq_lex.to_csv(index=False).encode("utf-8"),
                file_name="occurrences_consequences.csv",
                mime="text/csv",
                key=f"{key_prefix}dl_occ_consq_csv",
            )
    with colY:
        _subheader("Déclencheurs de cause (Regex)")
        if not use_regex_cc:
            st.info("Méthode Regex désactivée (voir barre latérale).")
        elif df_causes_lex.empty:
            st.info("Aucun déclencheur de cause détecté par Regex.")
        else:
            st.dataframe(df_causes_lex)
            st.download_button(
                "Exporter causes (CSV)",
                data=df_causes_lex.to_csv(index=False).encode("utf-8"),
                file_name="occurrences_causes.csv",
                mime="text/csv",
                key=f"{key_prefix}dl_occ_causes_csv",
            )
    _subheader("Texte annoté")

    codes_disponibles = sorted({str(v).upper() for v in dico_connecteurs.values()})
    show_codes: Dict[str, bool] = {}
    if codes_disponibles:
        st.markdown("**Familles de Connecteurs logiques**")
        col_all, col_none = st.columns(2)
        activer_conn = col_all.button(
            "Tout cocher (Connecteurs logiques)", key=f"{key_prefix}btn_conn_all"
        )
        desactiver_conn = col_none.button(
            "Tout décocher (Connecteurs logiques)", key=f"{key_prefix}btn_conn_none"
        )
        if activer_conn or desactiver_conn:
            toggle_checkboxes(
                f"{key_prefix}chk_code_",
                [code.lower() for code in codes_disponibles],
                activer_conn,
            )
        for code in codes_disponibles:
            show_codes[code] = st.checkbox(
                code,
                value=st.session_state.get(f"{key_prefix}chk_code_{code.lower()}", True),
                key=f"{key_prefix}chk_code_{code.lower()}",
            )

    categories_normatives = sorted({str(v).upper() for v in dico_marqueurs.values()})
    show_marqueurs_categories: Dict[str, bool] = {}
    if categories_normatives:
        st.markdown("**Marqueurs normatifs**")
        col_all, col_none = st.columns(2)
        activer_marqueurs = col_all.button(
            "Tout cocher (marqueurs)", key=f"{key_prefix}btn_marqueur_all"
        )
        desactiver_marqueurs = col_none.button(
            "Tout décocher (marqueurs)", key=f"{key_prefix}btn_marqueur_none"
        )
        if activer_marqueurs or desactiver_marqueurs:
            toggle_checkboxes(
                f"{key_prefix}chk_marqueur_",
                [cat.lower() for cat in categories_normatives],
                activer_marqueurs,
            )
        for cat in categories_normatives:
            label = cat.replace("_", " ")
            show_marqueurs_categories[cat] = st.checkbox(
                label,
                value=st.session_state.get(f"{key_prefix}chk_marqueur_{cat.lower()}", True),
                key=f"{key_prefix}chk_marqueur_{cat.lower()}",
            )
    else:
        show_marqueurs_categories = None

    categories_memoires = sorted({str(v).upper() for v in dico_memoires.values()})
    show_memoires_categories: Dict[str, bool] = {}
    if categories_memoires:
        st.markdown("**Marqueurs mémoire**")
        col_all, col_none = st.columns(2)
        activer_memoires = col_all.button(
            "Tout cocher (mémoire)", key=f"{key_prefix}btn_memoire_all"
        )
        desactiver_memoires = col_none.button(
            "Tout décocher (mémoire)", key=f"{key_prefix}btn_memoire_none"
        )
        if activer_memoires or desactiver_memoires:
            toggle_checkboxes(
                f"{key_prefix}chk_memoire_",
                [cat.lower() for cat in categories_memoires],
                activer_memoires,
            )
        for cat in categories_memoires:
            label = cat.replace("_", " ")
            show_memoires_categories[cat] = st.checkbox(
                label,
                value=st.session_state.get(f"{key_prefix}chk_memoire_{cat.lower()}", True),
                key=f"{key_prefix}chk_memoire_{cat.lower()}",
            )
    else:
        show_memoires_categories = None

    categories_tensions = sorted({str(v).upper() for v in dico_tensions.values()})
    show_tensions_categories: Optional[Dict[str, bool]] = None
    if categories_tensions:
        st.markdown("**Tensions sémantiques**")
        show_tensions = st.checkbox(
            "Afficher les tensions sémantiques",
            value=True,
            key=f"{key_prefix}chk_tensions_global",
        )
    else:
        show_tensions = False

    st.markdown("**Marqueurs de causalité**")
    show_consequences = st.checkbox(
        "CONSEQUENCE", value=True, key=f"{key_prefix}chk_consequence"
    )
    show_causes = st.checkbox("CAUSE", value=True, key=f"{key_prefix}chk_cause")

    st.markdown(css_badges(), unsafe_allow_html=True)
    if not texte_source.strip():
        st.info("Aucun texte fourni.")
        frag = "<div class='texte-annote'>(Texte vide)</div>"
    else:
        frag = html_annote(
            texte_source,
            dico_connecteurs,
            dico_marqueurs,
            dico_memoires,
            dico_consq,
            dico_causes,
            dico_tensions,
            show_codes,
            show_consequences,
            show_causes,
            show_tensions,
            show_marqueurs_categories=show_marqueurs_categories,
            show_memoires_categories=show_memoires_categories,
            show_tensions_categories=show_tensions_categories,
        )

    st.markdown(frag, unsafe_allow_html=True)
    st.download_button(
        "Exporter le texte annoté (HTML)",
        data=html_autonome(frag).encode("utf-8"),
        file_name="texte_annote.html",
        mime="text/html",
        key=f"{key_prefix}dl_html_annote",
    )


def render_analyses_tab(
    libelle_discours: str,
    texte_source: str,
    detections: Dict[str, pd.DataFrame],
    *,
    use_regex_cc: bool,
    dico_connecteurs: Dict[str, str],
    dico_marqueurs: Dict[str, str],
    dico_memoires: Dict[str, str],
    dico_consq: Dict[str, str],
    dico_causes: Dict[str, str],
    dico_tensions: Dict[str, str],
    key_prefix: str = "disc1_",
) -> None:
    st.markdown(f"### {libelle_discours}")
    render_detection_section(
        texte_source,
        detections,
        key_prefix=key_prefix,
        use_regex_cc=use_regex_cc,
        dico_connecteurs=dico_connecteurs,
        dico_marqueurs=dico_marqueurs,
        dico_memoires=dico_memoires,
        dico_consq=dico_consq,
        dico_causes=dico_causes,
        dico_tensions=dico_tensions,
    )

