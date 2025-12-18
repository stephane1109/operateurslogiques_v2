# -*- coding: utf-8 -*-
# main.py — Discours → Code (SI / ALORS / SINON / TANT QUE) + Marqueurs + Causes/Conséquences
# Ce script vise à détecter des connecteurs logiques et des marqueurs spécifiques dans le discours.
# Il est d’autant plus pertinent lorsqu’il est appliqué à plusieurs discours d’un même auteur ou locuteur prononcés dans des contextes différents.
# Méthodes comparées : Regex vs spaCy (modèle moyen si disponible)
#
# Fichiers requis dans le sous-répertoire dictionnaires/ :
#   - conditions.json        : mapping des segments conditionnels → CONDITION / ALORS / WHILE
#   - alternatives.json      : déclencheurs d’alternative → "ALTERNATIVE"
#   - dict_marqueurs.json    : marqueurs normatifs (OBLIGATION/INTERDICTION/…)
#   - consequences.json      : déclencheurs de conséquence → "CONSEQUENCE"
#   - causes.json            : déclencheurs de cause → "CAUSE"
#   - souvenirs.json         : déclencheurs liés à la mémoire → « MEM_* »
#
# Remarques :
#   - L’extraction CAUSE→CONSEQUENCE spaCy exploite la dépendance/les ancres causales et consécutives.
#   - Négation « ne … pouvoir … (pas/plus/jamais) » : ajustement par regex (sans options supplémentaires).
#   - Graphes conditionnels (IF/SI) et WHILE : rendu DOT (à l’écran) + export JPEG si Graphviz est présent (binaire 'dot').

import os
import re
import json
import html
import copy
from pathlib import Path
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Any, Optional

from analyses import (
    COULEURS_BADGES,
    COULEURS_MARQUEURS,
    COULEURS_TENSIONS,
    construire_regex_depuis_liste,
    render_analyses_tab,
    render_detection_section,
)

from import_exp import dictionnaire_to_bytes, parse_uploaded_dictionary

from stats import render_stats_tab
from stats_norm import render_stats_norm_tab
from conditions_spacy import analyser_conditions_spacy
from argToulmin import render_toulmin_tab
from lexique import render_lexique_tab
from storytelling.pynarrative import generer_storytelling_mermaid
from storytelling.actanciel import (
    analyser_actants_texte,
    charger_dictionnaire_actanciel,
    construire_tableau_actanciel,
    synthese_roles_actanciels,
)
from storytelling.sentiments import render_sentiments_tab
from storytelling.feel import render_feel_tab
from streamlit_utils import dataframe_safe
from text_utils import normaliser_espace, segmenter_en_phrases
from annotations import render_annotation_tab
from analaysesentiments import (
    render_camembert_tab,
    render_toxicite_tab,
    render_zero_shot_tab,
)

BASE_DIR = Path(__file__).resolve().parent
DICTIONNAIRES_DIR = BASE_DIR / "dictionnaires"

# =========================
# Détection Graphviz (pour export JPEG)
# =========================
try:
    import graphviz
    GV_OK = True
except Exception:
    GV_OK = False

def rendre_jpeg_depuis_dot(dot_str: str) -> bytes:
    """Rend en JPEG via graphviz.Source.pipe(format='jpg')."""
    if not GV_OK:
        raise RuntimeError("Graphviz (binaire 'dot') indisponible sur ce système.")
    src = graphviz.Source(dot_str)
    return src.pipe(format="jpg")

# =========================
# Chargement spaCy (modèles FR standards)
# =========================
SPACY_OK = False
NLP = None
SPACY_STATUS: List[str] = []
try:
    import spacy

    def _charger_modele_spacy(nom_modele: str) -> Any:
        """Tente de charger un modèle spaCy FR sans téléchargement automatique."""
        try:
            return spacy.load(nom_modele)
        except OSError:
            SPACY_STATUS.append(
                f"Modèle spaCy '{nom_modele}' absent. Installez-le manuellement"
                f" (ex.: python -m spacy download {nom_modele}) pour activer l'analyse NLP."
            )
        except Exception as err:
            SPACY_STATUS.append(
                f"Chargement du modèle spaCy '{nom_modele}' impossible : {err}"
            )
        return None

    for name in ("fr_core_news_md", "fr_core_news_sm"):
        modele = _charger_modele_spacy(name)
        if modele is not None:
            NLP = modele
            SPACY_OK = True
            SPACY_STATUS.append(f"Modèle spaCy chargé : {name}")
            if name == "fr_core_news_sm":
                SPACY_STATUS.append(
                    "Le modèle moyen 'fr_core_news_md' est recommandé pour de meilleures analyses."
                )
            break
    if not SPACY_OK:
        SPACY_STATUS.append("Aucun modèle spaCy FR n'a pu être chargé.")
except Exception as err:
    SPACY_OK = False
    NLP = None
    SPACY_STATUS.append(f"Import de spaCy impossible : {err}")

def _est_debut_segment(texte: str, index: int) -> bool:
    """Vérifie qu’un index correspond au début d’un segment (début ou précédé d’une ponctuation forte)."""
    if index <= 0:
        return True
    j = index - 1
    while j >= 0 and texte[j].isspace():
        j -= 1
    if j < 0:
        return True
    return texte[j] in ".,;:!?-—(«»\"'“”"

def _trouver_occurrences_motifs(texte: str, motifs: List[Tuple[str, re.Pattern]]) -> List[Dict[str, Any]]:
    """Retourne les occurrences non chevauchantes pour une liste de motifs (triée par position)."""
    if not texte or not motifs:
        return []
    occurrences: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for expr, pattern in motifs:
        for m in pattern.finditer(texte):
            span = (m.start(), m.end())
            if any(not (span[1] <= s[0] or span[0] >= s[1]) for s in spans):
                continue
            occurrences.append({
                "start": m.start(),
                "end": m.end(),
                "expression": expr,
                "match": texte[m.start():m.end()],
            })
            spans.append(span)
    occurrences.sort(key=lambda occ: occ["start"])
    return occurrences


def construire_df_phrases_storytelling(
    texte: str, detections: Dict[str, pd.DataFrame], libelle_discours: str
) -> pd.DataFrame:
    """Construit un DataFrame par phrase pour le module de storytelling."""

    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm) if texte_norm else []
    return pd.DataFrame(
        {
            "id_phrase": list(range(1, len(phrases) + 1)),
            "texte_phrase": phrases,
            "discours": libelle_discours,
        }
    )


def preparer_detections(texte_source: str, use_regex_cc: bool) -> Dict[str, pd.DataFrame]:
    """Retourne l'ensemble des DataFrames de détection pour un texte donné."""
    if texte_source.strip():
        df_conn = detecter_connecteurs(texte_source, DICO_CONNECTEURS)
        df_marq_brut = detecter_marqueurs(texte_source, DICO_MARQUEURS)
        df_marq = ajuster_negations_global(texte_source, df_marq_brut)
        df_memoires = detecter_memoires(texte_source, DICO_MEMOIRES)
        df_consq_lex = (
            detecter_consequences_lex(texte_source, DICO_CONSQS)
            if use_regex_cc
            else pd.DataFrame()
        )
        df_causes_lex = (
            detecter_causes_lex(texte_source, DICO_CAUSES)
            if use_regex_cc
            else pd.DataFrame()
        )
        df_tensions = detecter_tensions_semantiques(texte_source, DICO_TENSIONS)
    else:
        df_conn = pd.DataFrame()
        df_marq = pd.DataFrame()
        df_memoires = pd.DataFrame()
        df_consq_lex = pd.DataFrame()
        df_causes_lex = pd.DataFrame()
        df_tensions = pd.DataFrame()

    return {
        "df_conn": df_conn,
        "df_marq": df_marq,
        "df_memoires": df_memoires,
        "df_consq_lex": df_consq_lex,
        "df_causes_lex": df_causes_lex,
        "df_tensions": df_tensions,
    }

def _premier_match_motifs(
    texte: str,
    motifs: List[Tuple[str, re.Pattern]],
    start: int = 0,
    require_boundary: bool = False,
    skip_short: bool = False,
) -> Optional[re.Match]:
    """Renvoie le premier match satisfaisant les contraintes (ou None)."""
    if not texte or not motifs:
        return None
    meilleur: Optional[re.Match] = None
    for expr, pattern in motifs:
        expr_clean = expr.replace(" ", "")
        if skip_short and len(expr_clean) <= 2:
            continue
        m = pattern.search(texte, pos=start)
        if not m:
            continue
        if require_boundary and not _est_debut_segment(texte, m.start()):
            continue
        if meilleur is None or m.start() < meilleur.start():
            meilleur = m
    return meilleur

def _fin_clause_condition(phrase: str, start_pos: int) -> int:
    """Retourne l’index de la ponctuation suivant la condition (ou la fin de la phrase)."""
    if start_pos >= len(phrase):
        return len(phrase)
    sub = phrase[start_pos:]
    m = re.search(r"[,;:\-\—]\s+", sub)
    if m:
        return start_pos + m.start()
    return len(phrase)

def _extraire_condition_contenu(phrase: str, match_end: int, next_start: Optional[int]) -> str:
    """Extrait le contenu conditionnel après le déclencheur jusqu’à la ponctuation ou au prochain déclencheur."""
    segment = phrase[match_end:]
    if not segment:
        return ""
    limite = len(segment)
    m = re.search(r"[,;:\-\—]\s+", segment)
    if m:
        limite = min(limite, m.start())
    if next_start is not None:
        limite = min(limite, max(0, next_start - match_end))
    return segment[:limite].strip(" .;:-—")

def _expressions_par_etiquette(dico: Dict[str, str], etiquette: str) -> List[str]:
    """Filtre les expressions d’un dictionnaire selon leur étiquette normalisée."""
    cible = etiquette.upper()
    return [k for k, v in dico.items() if str(v).upper() == cible]

# =========================
# Chargement des dictionnaires JSON (dossier dictionnaires/)
# =========================
def charger_json_dico(chemin: os.PathLike[str] | str) -> Dict[str, str]:
    """Charge un JSON dict { expression: etiquette } ; normalise les clés côté détection."""
    path = Path(chemin)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Format JSON non supporté (attendu dict) : {path}")
    return {normaliser_espace(k.lower()): str(v).upper() for k, v in data.items() if k and str(k).strip()}

def charger_dicos_conditions() -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
]:
    """Charge les dictionnaires nécessaires au modèle SI / ALORS / SINON."""
    if not DICTIONNAIRES_DIR.is_dir():
        raise FileNotFoundError(
            f"Répertoire introuvable : {DICTIONNAIRES_DIR}"
        )
    d_cond = charger_json_dico(DICTIONNAIRES_DIR / "conditions.json")
    d_alt = charger_json_dico(DICTIONNAIRES_DIR / "alternatives.json")
    d_marq = charger_json_dico(DICTIONNAIRES_DIR / "dict_marqueurs.json")
    d_cons = charger_json_dico(DICTIONNAIRES_DIR / "consequences.json")
    d_caus = charger_json_dico(DICTIONNAIRES_DIR / "causes.json")
    d_mem = charger_json_dico(DICTIONNAIRES_DIR / "souvenirs.json")
    d_tension = charger_json_dico(DICTIONNAIRES_DIR / "tension_semantique.json")
    return d_cond, d_alt, d_marq, d_cons, d_caus, d_mem, d_tension

# =========================
# I/O discours
# =========================
def lire_fichier_txt(uploaded_file) -> str:
    """Lit un fichier .txt avec stratégie automatique d’encodage."""
    if uploaded_file is None:
        return ""
    donnees = uploaded_file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return donnees.decode(enc)
        except Exception:
            continue
    return donnees.decode("utf-8", errors="ignore")

# =========================
# Détection (Connecteurs logiques / marqueurs / conséquence / cause)
# =========================
def detecter_par_dico(texte: str, dico: Dict[str, str], champ_cle: str, champ_cat: str) -> pd.DataFrame:
    """Détection générique par dictionnaire clé→étiquette (insensible à la casse)."""
    if not dico:
        return pd.DataFrame()
    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm)
    motifs = construire_regex_depuis_liste(list(dico.keys()))
    enregs = []
    for i, ph in enumerate(phrases, start=1):
        ph_norm = normaliser_espace(ph)
        for cle_norm, motif in motifs:
            for m in motif.finditer(ph_norm):
                enregs.append({
                    "id_phrase": i,
                    "phrase": ph.strip(),
                    champ_cle: cle_norm,
                    champ_cat: dico[cle_norm],
                    "position": m.start(),
                    "longueur": m.end() - m.start()
                })
    df = pd.DataFrame(enregs)
    if not df.empty:
        df.sort_values(by=["id_phrase", "position"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    return df

def detecter_connecteurs(texte: str, dico_conn: Dict[str, str]) -> pd.DataFrame:
    return detecter_par_dico(texte, dico_conn, "connecteur", "code")

def detecter_marqueurs(texte: str, dico_marq: Dict[str, str]) -> pd.DataFrame:
    return detecter_par_dico(texte, dico_marq, "marqueur", "categorie")

def detecter_memoires(texte: str, dico_mem: Dict[str, str]) -> pd.DataFrame:
    return detecter_par_dico(texte, dico_mem, "memoire", "categorie")

def detecter_consequences_lex(texte: str, dico_consq: Dict[str, str]) -> pd.DataFrame:
    # On ajoute une colonne 'consequence' pour homogénéiser les affichages regex
    df = detecter_par_dico(texte, dico_consq, "consequence", "categorie")
    return df

def detecter_causes_lex(texte: str, dico_causes: Dict[str, str]) -> pd.DataFrame:
    # On ajoute une colonne 'cause' pour homogénéiser les affichages regex
    df = detecter_par_dico(texte, dico_causes, "cause", "categorie")
    return df

def detecter_tensions_semantiques(texte: str, dico_tensions: Dict[str, str]) -> pd.DataFrame:
    """Détection dédiée aux tensions sémantiques (dictionnaire simple)."""
    return detecter_par_dico(texte, dico_tensions, "expression", "tension")

# =========================
# Négation (regex autour de "pouvoir")
# =========================
def ajuster_negations_regex(phrase: str, dets_phrase: pd.DataFrame) -> pd.DataFrame:
    """
    Reclasse en INTERDICTION les formes 'ne … peut/peuvent/pourra/pourront (pas/plus/jamais)'.
    Ne traite pas d'autre verbe ; pas d'option 'ne … peut' sans 'pas'.
    """
    if dets_phrase.empty:
        return dets_phrase

    patrons = [
        r"\bne\s+\w{0,2}\s*peut\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
        r"\bne\s+\w{0,2}\s*peuvent\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
        r"\bne\s+\w{0,2}\s*pourra\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
        r"\bne\s+\w{0,2}\s*pourront\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
    ]
    spans_neg = []
    for pat in patrons:
        for m in re.finditer(pat, phrase, flags=re.I):
            spans_neg.append((m.start(), m.end()))

    if not spans_neg:
        return dets_phrase

    def chevauche(pos, longu, spans):
        debut, fin = pos, pos + longu
        for s, e in spans:
            if not (fin < s or debut > e):
                return True
        return False

    dets = dets_phrase.copy()
    cible = dets["marqueur"].str.lower().isin(["peut", "peuvent", "pourra", "pourront", "il peut"])
    for idx, row in dets.loc[cible].iterrows():
        if chevauche(row["position"], row["longueur"], spans_neg):
            dets.at[idx, "categorie"] = "INTERDICTION"
            if row["marqueur"].lower() == "il peut":
                dets.at[idx, "marqueur"] = "il ne peut (négation)"
            else:
                dets.at[idx, "marqueur"] = f"ne {row['marqueur']} (négation)"
    return dets

def ajuster_negations_global(texte: str, df_marq: pd.DataFrame) -> pd.DataFrame:
    """Applique les ajustements de négation phrase par phrase (regex)."""
    if df_marq.empty:
        return df_marq
    phrases = segmenter_en_phrases(texte)
    dets_list = []
    for i, ph in enumerate(phrases, start=1):
        bloc = df_marq[df_marq["id_phrase"] == i].copy()
        if bloc.empty:
            dets_list.append(bloc); continue
        bloc = ajuster_negations_regex(ph, bloc)
        dets_list.append(bloc)
    df_adj = pd.concat(dets_list, ignore_index=True) if dets_list else df_marq
    if not df_adj.empty:
        df_adj.sort_values(by=["id_phrase","position"], inplace=True, kind="mergesort")
        df_adj.reset_index(drop=True, inplace=True)
    return df_adj

# =========================
# Annotation HTML (texte + badges)
# =========================
def css_checkboxes_alignment() -> str:
    """CSS global pour harmoniser l'alignement des cases à cocher."""
    return """<style>
div[data-testid=\"stCheckbox\"] {
    display: flex;
    align-items: center;
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
}
div[data-testid=\"stCheckbox\"] > label {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    width: 100%;
}
div[data-testid=\"stCheckbox\"] > label div[data-testid=\"stMarkdownContainer\"] p {
    margin-bottom: 0;
}
</style>"""


# =========================
# Extraction graphes WHILE / IF (heuristiques)
# =========================
def extraire_segments_while(texte: str) -> List[Dict[str, Any]]:
    """Extrait 'tant que ...' ; condition = segment après 'tant que' jusqu’à la 1re ponctuation, action = suite éventuelle."""
    res = []
    phrases = segmenter_en_phrases(texte)
    for idx, ph in enumerate(phrases, start=1):
        for m in re.finditer(r"\btant\s+que\b", ph, flags=re.I):
            suite = ph[m.end():].strip()
            cut = re.split(r"[,;:\-\—]\s+", suite, maxsplit=1)
            condition = cut[0].strip() if cut and cut[0] else suite
            action = cut[1].strip() if cut and len(cut) > 1 else ""
            res.append({
                "type": "WHILE",
                "id_phrase": idx,
                "phrase": ph.strip(),
                "condition": condition,
                "action_true": action,
                "action_false": "",
                "debut": m.start(),
            })
    return res

def extraire_segments_if(texte: str) -> List[Dict[str, Any]]:
    """Extrait les structures conditionnelles « si … alors / sinon » en combinant phrases adjacentes."""
    if not texte or not COND_TERMS:
        return []

    COND_PAT = r"|".join(re.escape(x) for x in sorted(COND_TERMS, key=len, reverse=True))
    ALORS_PAT = r"|".join(re.escape(x) for x in sorted(ALORS_TERMS, key=len, reverse=True))
    ALT_PAT = r"|".join(re.escape(x) for x in sorted(ALT_TERMS, key=len, reverse=True))
    WH_PAT = (
        r"|".join(re.escape(x) for x in sorted(WHILE_TERMS, key=len, reverse=True))
        if WHILE_TERMS else r"\btant\s+que\b"
    )

    alpha = "A-Za-zÀ-ÖØ-öø-ÿ"
    cond_re = re.compile(rf"(?<![{alpha}])(?:{COND_PAT})(?![{alpha}])", flags=re.I)
    alors_re = re.compile(rf"(?<![{alpha}])(?:{ALORS_PAT})(?![{alpha}])", flags=re.I) if ALORS_PAT else None
    alt_re = re.compile(rf"(?<![{alpha}])(?:{ALT_PAT})(?![{alpha}])", flags=re.I) if ALT_PAT else None
    alt_head_re = re.compile(rf"^\s*(?:{ALT_PAT})(?![{alpha}])", flags=re.I) if ALT_PAT else None
    alors_head_re = re.compile(rf"^\s*(?:{ALORS_PAT})(?![{alpha}])", flags=re.I) if ALORS_PAT else None
    while_re = re.compile(rf"(?<![{alpha}])(?:{WH_PAT})(?![{alpha}])", flags=re.I)

    segments: List[Dict[str, Any]] = []
    phrases = segmenter_en_phrases(texte)

    for idx, phrase in enumerate(phrases):
        phrase_id = idx + 1
        if not phrase.strip():
            continue

        while_spans = [(m.start(), m.end()) for m in while_re.finditer(phrase)] if WH_PAT else []

        cond_matches: List[Dict[str, Any]] = []
        for m in cond_re.finditer(phrase):
            if any(start <= m.start() < end for start, end in while_spans):
                continue
            cond_matches.append({
                "start": m.start(),
                "end": m.end(),
                "match": m.group(),
            })

        if not cond_matches:
            continue

        last_end = cond_matches[-1]["end"]
        reste = phrase[last_end:]
        strip_len = len(reste) - len(reste.lstrip(" ,;:-—"))
        apres_cond_start = last_end + strip_len

        alt_match_current = alt_re.search(phrase, apres_cond_start) if alt_re else None
        alors_match_current = alors_re.search(phrase, apres_cond_start) if alors_re else None

        action_true_phrase_index = idx
        action_true_start = apres_cond_start
        action_true_segments: List[str] = []
        action_false = ""

        if alors_match_current:
            action_true_start = alors_match_current.end()
        elif alors_head_re and idx + 1 < len(phrases):
            prochain = phrases[idx + 1]
            match_next = alors_head_re.match(prochain)
            if match_next:
                action_true_phrase_index = idx + 1
                action_true_start = match_next.end()
        if action_true_phrase_index != idx and apres_cond_start < len(phrase):
            limite = alt_match_current.start() if alt_match_current else len(phrase)
            if apres_cond_start < limite:
                segment = phrase[apres_cond_start:limite].strip(" .;:-—")
                if segment:
                    action_true_segments.append(segment)

        action_phrase = phrases[action_true_phrase_index]
        alt_match_action = alt_re.search(action_phrase, action_true_start) if alt_re else None

        if alt_match_action:
            action_segment = action_phrase[action_true_start:alt_match_action.start()].strip(" .;:-—")
            action_false = action_phrase[alt_match_action.end():].strip(" .;:-—")
        else:
            action_segment = action_phrase[action_true_start:].strip(" .;:-—")

        if action_segment:
            action_true_segments.append(action_segment)

        if not action_false and alt_head_re:
            suivant_index = action_true_phrase_index + 1
            if suivant_index < len(phrases):
                phrase_suiv = phrases[suivant_index]
                alt_suivant = alt_head_re.match(phrase_suiv) or (alt_re.search(phrase_suiv) if alt_re else None)
                if alt_suivant:
                    action_false = phrase_suiv[alt_suivant.end():].strip(" .;:-—")

        action_true = " ".join(part for part in action_true_segments if part).strip()

        for pos, match in enumerate(cond_matches):
            next_start: Optional[int] = None
            if pos + 1 < len(cond_matches):
                next_start = cond_matches[pos + 1]["start"]
            condition = _extraire_condition_contenu(phrase, match["end"], next_start)

            segments.append({
                "type": "IF",
                "id_phrase": phrase_id,
                "phrase": phrase.strip(),
                "condition": condition,
                "action_true": action_true,
                "action_false": action_false,
                "debut": match["start"],
            })

    return segments

def graphviz_while_dot(condition: str, action: str) -> str:
    """Construit un DOT simple pour WHILE."""
    def esc(s: str) -> str: return s.replace('"', r"\"")
    cond_txt = esc(condition if condition else "(condition non extraite)")
    act_txt = esc(action if action else "(action implicite ou non extraite)")
    return f'''
digraph G {{
  rankdir=LR;
  node [shape=box, fontname="Helvetica"];
  start [shape=circle, label="Start"];
  cond [shape=diamond, label="while ({cond_txt})"];
  act  [shape=box, label="{act_txt}"];
  end  [shape=doublecircle, label="End"];
  start -> cond;
  cond -> act [label="Vrai"];
  act  -> cond [label="itère"];
  cond -> end [label="Faux"];
}}
'''

def graphviz_if_dot(condition: str, action_true: str, action_false: str = "") -> str:
    """Construit un DOT simple pour IF."""
    def esc(s: str) -> str: return s.replace('"', r"\"")
    cond_txt = esc(condition if condition else "(condition non extraite)")
    act_t = esc(action_true if action_true else "(action si vrai non extraite)")
    act_f = esc(action_false) if action_false else ""
    has_else = bool(action_false.strip())
    lignes = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Helvetica"];',
        '  start [shape=circle, label="Start"];',
        f'  cond  [shape=diamond, label="if ({cond_txt})"];',
        f'  actt  [shape=box, label="Action si Vrai: {act_t}"];',
    ]
    if has_else:
        lignes.append(f'  actf  [shape=box, label="Action si Faux: {act_f}"];')
    lignes.append('  end   [shape=doublecircle, label="End"];')
    lignes += [
        "  start -> cond;",
        '  cond  -> actt [label="Vrai"];',
        "  actt  -> end;",
    ]
    if has_else:
        lignes += ['  cond  -> actf [label="Faux"];', "  actf  -> end;"]
    lignes.append("}")
    return "\n".join(lignes)

# =========================
# spaCy : extraction CAUSE → CONSÉQUENCE
# =========================
def _locution_match(tok, locutions_norm: set) -> bool:
    """Teste un match simple sur le token lui-même ou sa sous-chaîne de sous-arbre."""
    t = tok.lower_
    if t in locutions_norm:
        return True
    surface = " ".join(w.lower_ for w in tok.subtree)
    return any(loc in surface for loc in locutions_norm)

def extraire_cause_consequence_spacy(texte: str, nlp, causes_lex: List[str], consequences_lex: List[str]) -> pd.DataFrame:
    """
    Retourne un DataFrame avec les segments CAUSE/CONSÉQUENCE extraits par analyse dépendancielle.
    Colonnes : id_phrase, type, cause_span, consequence_span, ancre, methode, phrase
    """
    if not nlp:
        return pd.DataFrame()

    doc = nlp(texte)
    causes_norm = {c.lower().strip() for c in causes_lex}
    consq_norm = {c.lower().strip() for c in consequences_lex}
    enregs = []
    prev_sent_text = ""

    for pid, sent in enumerate(doc.sents, start=1):
        # Subordonnées causales (mark ∈ causes)
        for tok in sent:
            if tok.dep_.lower() == "mark" and _locution_match(tok, causes_norm):
                head = tok.head
                cause_span = doc[head.left_edge.i: head.right_edge.i+1]
                enregs.append({
                    "id_phrase": pid,
                    "type": "CAUSE_SUBORDONNEE",
                    "cause_span": cause_span.text,
                    "consequence_span": sent.text,
                    "ancre": tok.text,
                    "methode": "mark→advcl",
                    "phrase": sent.text
                })

        # Groupes prépositionnels causaux (à cause de, en raison de, du fait de, grâce à…)
        for tok in sent:
            if tok.dep_.lower() in {"case","fixed","mark"} and _locution_match(tok, causes_norm):
                head = tok.head
                gn = doc[head.left_edge.i: head.right_edge.i+1]
                enregs.append({
                    "id_phrase": pid,
                    "type": "CAUSE_GN",
                    "cause_span": gn.text,
                    "consequence_span": sent.text,
                    "ancre": tok.text,
                    "methode": "case/fixed→obl",
                    "phrase": sent.text
                })

        # Conséquence adverbiale en tête de phrase (donc, alors, ainsi, dès lors…)
        premiers = [t for t in sent if not t.is_punct][:3]
        if premiers:
            t0 = premiers[0]
            if _locution_match(t0, consq_norm) and t0.pos_ in {"ADV","CCONJ","SCONJ"}:
                enregs.append({
                    "id_phrase": pid,
                    "type": "CONSEQUENCE_ADV",
                    "cause_span": prev_sent_text,
                    "consequence_span": sent.text,
                    "ancre": t0.text,
                    "methode": "adv/discourse tête de phrase",
                    "phrase": sent.text
                })

        # Subordonnées consécutives (de sorte que, si bien que, de façon que…)
        for tok in sent:
            if tok.dep_.lower() == "mark" and _locution_match(tok, consq_norm):
                head = tok.head
                cons_span = doc[head.left_edge.i: head.right_edge.i+1]
                enregs.append({
                    "id_phrase": pid,
                    "type": "CONSEQUENCE_SUBORDONNEE",
                    "cause_span": sent.text,
                    "consequence_span": cons_span.text,
                    "ancre": tok.text,
                    "methode": "mark→advcl(consécutif)",
                    "phrase": sent.text
                })

        prev_sent_text = sent.text

    df = pd.DataFrame(enregs)
    if not df.empty:
        df = df.sort_values(["id_phrase", "type"]).reset_index(drop=True)
    return df

# =========================
# Helpers pour tableaux comparatifs (surlignage ⟦ … ⟧)
# =========================
def marquer_terme_brut(phrase: str, terme: str) -> str:
    """Entoure la 1ère occurrence de 'terme' par ⟦…⟧ (insensible à la casse), sans HTML."""
    if not phrase or not terme:
        return phrase or ""
    m = re.search(re.escape(terme), phrase, flags=re.I)
    if not m:
        return phrase
    i, j = m.start(), m.end()
    return phrase[:i] + "⟦" + phrase[i:j] + "⟧" + phrase[j:]

def table_regex_df(df: pd.DataFrame, type_marqueur: str) -> pd.DataFrame:
    """
    Construit un DataFrame pour l’affichage Streamlit :
      - type_marqueur = "CAUSE"  -> colonne clé = 'cause'
      - type_marqueur = "CONSEQUENCE" -> colonne clé = 'consequence'
    Ajoute 'phrase_marquee' avec le marqueur entouré de ⟦…⟧.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["id_phrase", "marqueur", "catégorie", "phrase_marquee"])

    cle = "cause" if type_marqueur.upper() == "CAUSE" else "consequence"
    lignes = []
    for _, row in df.iterrows():
        marqueur = row.get(cle, "")
        cat = row.get("categorie", "")
        phr = row.get("phrase", "")
        phr_m = marquer_terme_brut(phr, marqueur)
        lignes.append({
            "id_phrase": row.get("id_phrase", ""),
            "marqueur": marqueur,
            "catégorie": cat,
            "phrase_marquee": phr_m
        })
    return pd.DataFrame(lignes)

def table_spacy_df(df_spacy: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DataFrame pour l’affichage Streamlit côté spaCy :
    Colonnes : id_phrase, type, ancre, méthode, phrase_marquee, cause_span, consequence_span
    (l’ancre est entourée de ⟦…⟧ dans la phrase).
    """
    if df_spacy is None or df_spacy.empty:
        return pd.DataFrame(columns=["id_phrase", "type", "ancre", "méthode", "phrase_marquee", "cause_span", "consequence_span"])

    lignes = []
    for _, row in df_spacy.iterrows():
        ancre = row.get("ancre", "")
        phr = row.get("phrase", "")
        phr_m = marquer_terme_brut(phr, ancre)
        lignes.append({
            "id_phrase": row.get("id_phrase", ""),
            "type": row.get("type", ""),
            "ancre": ancre,
            "méthode": row.get("methode", ""),
            "phrase_marquee": phr_m,
            "cause_span": row.get("cause_span", ""),
            "consequence_span": row.get("consequence_span", "")
        })
    return pd.DataFrame(lignes)

# =========================
# Interface Streamlit
# =========================
st.set_page_config(
    page_title="Formalisation logique du langage naturel (connecteurs : Si / Alors / Sinon / Tant que + marqueurs + causes/conséquences",
    page_icon=None,
    layout="wide",
)
st.markdown(css_checkboxes_alignment(), unsafe_allow_html=True)
st.title(
    "Formalisation logique du langage naturel (connecteurs : Si / Alors / Sinon / Tant que + marqueurs + causes/conséquences"
)
st.caption(
    "Vous pouvez récupérer deux fichiers texte (Discours de Politique Générale de Sébastien "
    "Lecornu devant l'Assemblée Nationale et le Sénat) pour tester l'interface : "
    "[répertoire exemples](https://github.com/stephane1109/if_else_discours/tree/main/exemples)."
)
st.markdown("[www.codeandcortex.fr](https://www.codeandcortex.fr)")
st.write("")

# Chargement des dicos
try:
    (
        DICO_CONDITIONS,
        DICO_ALTERNATIVES,
        DICO_MARQUEURS,
        DICO_CONSQS,
        DICO_CAUSES,
        DICO_MEMOIRES,
        DICO_TENSIONS,
    ) = charger_dicos_conditions()
    DICO_ACTANCIEL = charger_dictionnaire_actanciel(DICTIONNAIRES_DIR / "actanciel.py")
except Exception as e:
    st.error("Impossible de charger les dictionnaires JSON depuis le dossier 'dictionnaires/'.")
    st.code(str(e))
    st.stop()

DICOS_REFERENCE = {
    "conditions": copy.deepcopy(DICO_CONDITIONS),
    "alternatives": copy.deepcopy(DICO_ALTERNATIVES),
    "marqueurs": copy.deepcopy(DICO_MARQUEURS),
    "consequences": copy.deepcopy(DICO_CONSQS),
    "causes": copy.deepcopy(DICO_CAUSES),
    "souvenirs": copy.deepcopy(DICO_MEMOIRES),
    "tensions": copy.deepcopy(DICO_TENSIONS),
}

# Session State : dictionnaires actifs (peuvent être remplacés par un import JSON)
if "dicos_actifs" not in st.session_state:
    st.session_state["dicos_actifs"] = copy.deepcopy(DICOS_REFERENCE)

dicos_actifs = st.session_state["dicos_actifs"]
DICO_CONDITIONS = dicos_actifs.get("conditions", {})
DICO_ALTERNATIVES = dicos_actifs.get("alternatives", {})
DICO_MARQUEURS = dicos_actifs.get("marqueurs", {})
DICO_CONSQS = dicos_actifs.get("consequences", {})
DICO_CAUSES = dicos_actifs.get("causes", {})
DICO_MEMOIRES = dicos_actifs.get("souvenirs", {})
DICO_TENSIONS = dicos_actifs.get("tensions", {})

DICO_CONNECTEURS: Dict[str, str] = {**DICO_CONDITIONS, **DICO_ALTERNATIVES}

COND_TERMS = {k for k, v in DICO_CONDITIONS.items() if str(v).upper() == "CONDITION"}
ALORS_TERMS = {k for k, v in DICO_CONDITIONS.items() if str(v).upper() == "ALORS"}
WHILE_TERMS = {k for k, v in DICO_CONDITIONS.items() if str(v).upper() == "WHILE"}
ALT_TERMS = set(DICO_ALTERNATIVES.keys())

# Alerte spaCy/Graphviz
if not SPACY_OK:
    st.warning(
        "spaCy FR indisponible (installez par exemple le modèle 'fr_core_news_md'). L’onglet spaCy utilisera uniquement Regex si aucun modèle FR n’est chargé."
    )
if SPACY_STATUS:
    st.caption(" ; ".join(SPACY_STATUS))
if not GV_OK:
    st.warning("Graphviz non détecté : l’export JPEG des graphes ne sera pas disponible (rendu DOT affiché quand même).")

# Barre latérale : choix méthodes
with st.sidebar:
    st.header("Méthodes d’analyse")
    use_regex_cc = st.checkbox("Dictionnaire json (règles Regex)", value=True)
    use_spacy_dev_cc = st.checkbox("Dictionnaire NLP (SpaCy) - en cours de dév", value=False)
    if use_spacy_dev_cc:
        st.caption("Fonctionnalité spaCy en cours de développement.")

# La méthode spaCy principale est désactivée (case supprimée de l’interface)
use_spacy_cc = False

# Source du discours
st.markdown("### Source du discours")
mode_source = st.radio("Choisir la source du texte", ["Fichier .txt", "Zone de texte"], index=0, horizontal=True)

nom_discours_1 = "Discours 1"
nom_discours_2 = "Discours 2"

texte_source = ""
texte_source_2 = ""
if mode_source == "Fichier .txt":
    fichier_txt = st.file_uploader(
        "Déposer un fichier texte (.txt)",
        type=["txt"],
        accept_multiple_files=False,
        key="discours_txt",
    )
    fichier_txt_2 = st.file_uploader(
        "Déposer un deuxième fichier texte (.txt) pour comparer",
        type=["txt"],
        accept_multiple_files=False,
        key="discours_txt_2",
    )
    if fichier_txt is not None:
        try:
            texte_source = lire_fichier_txt(fichier_txt)
            nom_discours_1 = fichier_txt.name
            st.success(
                f"Fichier chargé : {fichier_txt.name} • {len(texte_source)} caractères"
            )
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
    if fichier_txt_2 is not None:
        try:
            texte_source_2 = lire_fichier_txt(fichier_txt_2)
            nom_discours_2 = fichier_txt_2.name
            st.success(
                f"Second fichier chargé : {fichier_txt_2.name} • {len(texte_source_2)} caractères"
            )
        except Exception as e:
            st.error(f"Impossible de lire le deuxième fichier : {e}")
else:
    texte_source = st.text_area(
        "Saisir ou coller le discours :",
        value="",
        height=240,
        placeholder="Coller ici le discours à analyser…",
    )
    texte_source_2 = st.text_area(
        "(Optionnel) Saisir ou coller un deuxième discours :",
        value="",
        height=240,
        placeholder="Coller ici le deuxième discours à comparer…",
    )

st.divider()

# Détections de base
detections_1 = preparer_detections(texte_source, use_regex_cc)
detections_2 = preparer_detections(texte_source_2, use_regex_cc)
df_conn = detections_1["df_conn"]
df_marq = detections_1["df_marq"]
df_memoires = detections_1["df_memoires"]
df_consq_lex = detections_1["df_consq_lex"]
df_causes_lex = detections_1["df_causes_lex"]
df_tensions = detections_1["df_tensions"]
df_conn_2 = detections_2["df_conn"]
df_marq_2 = detections_2["df_marq"]
df_memoires_2 = detections_2["df_memoires"]
df_consq_lex_2 = detections_2["df_consq_lex"]
df_causes_lex_2 = detections_2["df_causes_lex"]
df_tensions_2 = detections_2["df_tensions"]

couleur_discours_1 = "#c00000"
couleur_discours_2 = "#1f4e79"
libelle_discours_1 = (
    f"Discours 1 - {nom_discours_1}"
    if nom_discours_1.strip() and nom_discours_1.strip() != "Discours 1"
    else "Discours 1"
)
libelle_discours_2 = (
    f"Discours 2 - {nom_discours_2}"
    if nom_discours_2.strip() and nom_discours_2.strip() != "Discours 2"
    else "Discours 2"
)

# Onglets

(
    tab_detections,
    tab_conditions,
    tab_stats,
    tab_stats_norm,
    tab_discours,
    tab_toulmin,
    tab_dicos,
    tab_lexique,
    tab_annot,
    tab_storytelling,
    tab_sentiments,
    tab_camembert,
    tab_toxicite,
    tab_zero_shot,
    tab_feel,
) = st.tabs(
    [
        "Analyses",
        "conditions logiques : si/alors",
        "Stats",
        "Stats norm",
        "2 discours",
        "Arg Toulmin",
        "Dictionnaires (JSON)",
        "Lexique",
        "Annot",
        "Storytelling",
        "ASentsVader",
        "AnalysSentCamemBert",
        "AnalysSentToxic",
        "zeroclassification",
        "FEEL",
    ]
)

# Onglet désactivé : Comparatif règles Regex vs spaCy
tab_comparatif = None

# Onglet Analyses (listes + texte annoté)
with tab_detections:
    analyse_options = [
        {
            "id": "disc1",
            "label": libelle_discours_1,
            "texte": texte_source,
            "detections": detections_1,
        }
    ]
    if texte_source_2.strip() or (
        nom_discours_2.strip() and nom_discours_2.strip() != "Discours 2"
    ):
        analyse_options.append(
            {
                "id": "disc2",
                "label": libelle_discours_2,
                "texte": texte_source_2,
                "detections": detections_2,
            }
        )

    option_ids = [opt["id"] for opt in analyse_options]
    labels_by_id = {opt["id"]: opt["label"] for opt in analyse_options}
    selected_id = st.selectbox(
        "Choisir le discours à analyser",
        options=option_ids,
        format_func=lambda oid: labels_by_id.get(oid, oid),
    )
    selection_analyse = next(
        (opt for opt in analyse_options if opt["id"] == selected_id), analyse_options[0]
    )

    render_analyses_tab(
        selection_analyse["label"],
        selection_analyse["texte"],
        selection_analyse["detections"],
        use_regex_cc=use_regex_cc,
        dico_connecteurs=DICO_CONNECTEURS,
        dico_marqueurs=DICO_MARQUEURS,
        dico_memoires=DICO_MEMOIRES,
        dico_consq=DICO_CONSQS,
        dico_causes=DICO_CAUSES,
        dico_tensions=DICO_TENSIONS,
        key_prefix=f"{selection_analyse['id']}_",
    )

# Onglet Dictionnaires (JSON)
with tab_dicos:
    st.subheader("Aperçu des dictionnaires chargés (dossier 'dictionnaires/')")
    st.caption(
        "Vous pouvez importer votre propre dictionnaire JSON (répertoire dictionnaires/),"
        " mais vous devez impérativement respecter le nom d’origine."
    )

    def bloc_dictionnaire(
        titre: str,
        cle: str,
        dico_actif: Dict[str, str],
    ):
        st.markdown(f"**{titre}**")
        st.caption("Dictionnaire actif (lecture seule)")
        st.json(dico_actif, expanded=False)
        st.download_button(
            "Télécharger le dictionnaire actif",
            data=dictionnaire_to_bytes(dico_actif),
            file_name=f"{cle}_actif.json",
            mime="application/json",
            key=f"dl_actif_{cle}_json",
        )
        fichier_import = st.file_uploader(
            "Importer un dictionnaire JSON",
            type=["json"],
            key=f"upload_{cle}",
            label_visibility="visible",
            help="Le fichier doit contenir un objet JSON {expression: etiquette}.",
        )
        if fichier_import is not None:
            try:
                dico_charge = parse_uploaded_dictionary(
                    fichier_import, normalizer=normaliser_espace
                )
                st.session_state["dicos_actifs"][cle] = dico_charge
                st.success(
                    "Dictionnaire personnalisé chargé : il est maintenant utilisé pour les analyses."
                )
                # Le déclenchement explicite d'un rerun provoquait des boucles
                # infinies sur Streamlit Cloud (l'uploader renvoyant toujours
                # un fichier non vide). Le dictionnaire est déjà injecté dans
                # st.session_state ; on laisse Streamlit rafraîchir la page
                # normalement sans forcer un rerun manuel.
            except Exception as err:
                st.error(f"Import impossible : {err}")

    bloc_dictionnaire("conditions.json", "conditions", DICO_CONDITIONS)
    bloc_dictionnaire("alternatives.json", "alternatives", DICO_ALTERNATIVES)
    bloc_dictionnaire("dict_marqueurs.json", "marqueurs", DICO_MARQUEURS)
    bloc_dictionnaire("consequences.json", "consequences", DICO_CONSQS)
    bloc_dictionnaire("causes.json", "causes", DICO_CAUSES)
    bloc_dictionnaire("souvenirs.json", "souvenirs", DICO_MEMOIRES)
    bloc_dictionnaire("tension_semantique.json", "tensions", DICO_TENSIONS)

# Onglet Lexique
with tab_lexique:
    render_lexique_tab()

# Onglet Argumentation (Toulmin)
with tab_toulmin:
    render_toulmin_tab(
        texte_source,
        texte_source_2=texte_source_2,
        heading_discours_1=libelle_discours_1,
        heading_discours_2=libelle_discours_2,
        couleur_discours_1=couleur_discours_1,
        couleur_discours_2=couleur_discours_2,
    )

# Onglet 3 : conditions logiques : si/alors
with tab_conditions:
    st.subheader("Segments conditionnels détectés (SI / ALORS / SINON / TANT QUE)")
    titre_conditions_regex = (
        f"{libelle_discours_1} - sur la base des règles regex (json)"
    )
    st.markdown(
        f"<span style='color:{couleur_discours_1}; font-weight:600;'>{html.escape(titre_conditions_regex)}</span>",
        unsafe_allow_html=True,
    )
    if not texte_source.strip():
        st.info("Aucun texte fourni.")
    else:
        seg_while = extraire_segments_while(texte_source)
        seg_if = extraire_segments_if(texte_source)
        segments_conditionnels = sorted(
            seg_while + seg_if,
            key=lambda s: (s.get("id_phrase", 0), s.get("debut", 0))
        )

        if not segments_conditionnels:
            st.info("Aucun segment conditionnel détecté.")
        else:
            for idx_seg, sel in enumerate(segments_conditionnels, start=1):
                type_sel = str(sel.get("type", "")).upper()
                if type_sel == "WHILE":
                    st.markdown(f"**WHILE — phrase {sel['id_phrase']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("Condition")
                        st.write(sel["condition"] if sel["condition"] else "(non extraite)")
                    with col2:
                        st.markdown("Action (itération)")
                        st.write(sel["action_true"] if sel["action_true"] else "(implicite ou non extraite)")

                    dot = graphviz_while_dot(sel["condition"], sel["action_true"])
                    st.graphviz_chart(dot, use_container_width=True)

                    if GV_OK:
                        try:
                            img_bytes = rendre_jpeg_depuis_dot(dot)
                            st.download_button(
                                f"Télécharger le graphe WHILE #{sel['id_phrase']} (JPEG)",
                                data=img_bytes,
                                file_name=f"while_phrase_{sel['id_phrase']}.jpg",
                                mime="image/jpeg",
                                key=f"dl_while_jpg_{sel['id_phrase']}_{idx_seg}"
                            )
                        except Exception as e:
                            st.error(f"Export JPEG indisponible : {e}")
                else:
                    st.markdown(f"**IF — phrase {sel['id_phrase']}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("Condition (if)")
                        st.write(sel["condition"] if sel["condition"] else "(non extraite)")
                    with col2:
                        st.markdown("Action si Vrai (alors)")
                        st.write(sel["action_true"] if sel["action_true"] else "(implicite ou non extraite)")
                    with col3:
                        st.markdown("Action si Faux (else)")
                        st.write(sel["action_false"] if sel["action_false"] else "(absente)")

                    dot = graphviz_if_dot(sel["condition"], sel["action_true"], sel["action_false"])
                    st.graphviz_chart(dot, use_container_width=True)

                    if GV_OK:
                        try:
                            img_bytes = rendre_jpeg_depuis_dot(dot)
                            st.download_button(
                                f"Télécharger le graphe IF #{sel['id_phrase']} (JPEG)",
                                data=img_bytes,
                                file_name=f"if_phrase_{sel['id_phrase']}.jpg",
                                mime="image/jpeg",
                                key=f"dl_if_jpg_{sel['id_phrase']}_{idx_seg}"
                            )
                        except Exception as e:
                            st.error(f"Export JPEG indisponible : {e}")

                with st.expander("Voir la phrase complète"):
                    st.write(sel["phrase"])
                st.markdown("---")

        st.markdown("### Analyse spaCy (modèle fr_core_news_md)")
        if not SPACY_OK or NLP is None:
            st.info(
                "spaCy FR n'est pas disponible. Installez par exemple "
                "`fr_core_news_md` pour activer cette analyse."
            )
        else:
            df_conditions_spacy = analyser_conditions_spacy(
                texte_source,
                NLP,
                sorted(COND_TERMS),
                sorted(ALORS_TERMS),
                sorted(ALT_TERMS),
            )
            if df_conditions_spacy.empty:
                st.info(
                    "Aucune structure conditionnelle n'a été identifiée par spaCy."
                )
            else:
                dataframe_safe(
                    df_conditions_spacy,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Exporter l'analyse spaCy (CSV)",
                    data=df_conditions_spacy.to_csv(index=False).encode("utf-8"),
                    file_name="conditions_spacy.csv",
                    mime="text/csv",
                    key="dl_conditions_spacy_csv",
                )

# Onglet Comparatif Regex / spaCy (désactivé)
if tab_comparatif is not None:
    with tab_comparatif:
        st.subheader("Comparatif des détections Causes/Conséquences : Regex vs spaCy")

        if not texte_source.strip():
            st.info("Aucun texte fourni.")
        else:
            # 1) Regex — CAUSE
            st.markdown("**Détections Regex — CAUSE**")
            if not use_regex_cc:
                st.info("Méthode Regex désactivée (voir barre latérale).")
            else:
                if df_causes_lex.empty:
                    st.info("Aucune CAUSE trouvée par Regex.")
                else:
                    df_view_cause = table_regex_df(df_causes_lex, "CAUSE")
                    dataframe_safe(df_view_cause, use_container_width=True, hide_index=True)

            st.markdown("---")

            # 2) Regex — CONSEQUENCE
            st.markdown("**Détections Regex — CONSEQUENCE**")
            if not use_regex_cc:
                st.info("Méthode Regex désactivée (voir barre latérale).")
            else:
                if df_consq_lex.empty:
                    st.info("Aucune CONSEQUENCE trouvée par Regex.")
                else:
                    df_view_consq = table_regex_df(df_consq_lex, "CONSEQUENCE")
                    dataframe_safe(df_view_consq, use_container_width=True, hide_index=True)

            st.markdown("---")

            # 3) spaCy — CAUSE → CONSÉQUENCE
            st.markdown("**Détections spaCy — CAUSE → CONSÉQUENCE**")
            if use_spacy_cc and SPACY_OK and NLP is not None:
                df_cc_spacy = extraire_cause_consequence_spacy(
                    texte_source,
                    NLP,
                    list(DICO_CAUSES.keys()),
                    list(DICO_CONSQS.keys())
                )
                if df_cc_spacy.empty:
                    st.info("Aucun lien trouvé par spaCy (selon les ancres fournies).")
                else:
                    df_spacy_view = table_spacy_df(df_cc_spacy)
                    dataframe_safe(df_spacy_view, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Exporter CAUSE → CONSÉQUENCE (CSV)",
                        data=df_spacy_view.to_csv(index=False).encode("utf-8"),
                        file_name="cause_consequence_spacy.csv",
                        mime="text/csv",
                        key="dl_cc_spacy_csv"
                    )
            elif use_spacy_cc and not SPACY_OK:
                st.warning("spaCy FR indisponible (installez un modèle français, par exemple 'fr_core_news_md').")
            else:
                st.info("Analyse spaCy désactivée.")

# Onglet Annot (création de dictionnaires à partir de surlignages)
with tab_annot:
    render_annotation_tab(texte_source)

# Onglet 5 : Statistiques sur les marqueurs
with tab_stats:
    render_stats_tab(
        texte_source,
        df_conn,
        df_marq,
        df_memoires,
        df_consq_lex,
        df_causes_lex,
        df_tensions,
        texte_source_2=texte_source_2,
        df_conn_2=df_conn_2,
        df_marqueurs_2=df_marq_2,
        df_memoires_2=df_memoires_2,
        df_consq_lex_2=df_consq_lex_2,
        df_causes_lex_2=df_causes_lex_2,
        df_tensions_2=df_tensions_2,
        heading_discours_1=libelle_discours_1,
        heading_discours_2=libelle_discours_2,
        couleur_discours_1=couleur_discours_1,
        couleur_discours_2=couleur_discours_2,
    )

with tab_storytelling:
    st.subheader("Storytelling")
    st.caption(
        "Construction d'une narration par scènes à partir des phrases annotées, "
        "avec export du flowchart Mermaid prêt à être visualisé."
    )

    # Préparation des phrases annotées pour chaque discours
    df_phrases_1 = construire_df_phrases_storytelling(
        texte_source, detections_1, libelle_discours_1
    )
    df_phrases_2 = construire_df_phrases_storytelling(
        texte_source_2, detections_2, libelle_discours_2
    )

    options_df = {}
    if not df_phrases_1.empty:
        options_df[libelle_discours_1] = df_phrases_1
    if not df_phrases_2.empty:
        options_df[libelle_discours_2] = df_phrases_2

    if not options_df:
        st.info("Aucune phrase disponible pour construire le storytelling narratif.")
    else:
        choix_discours = st.selectbox(
            "Choisissez le discours à transformer en narration",
            options=list(options_df.keys()),
        )

        if st.button("Générer le storytelling narratif", key="btn_storytelling"):
            df_cible = options_df.get(choix_discours, pd.DataFrame())
            if df_cible.empty:
                st.warning("Aucune phrase détectée pour ce discours.")
            else:
                try:
                    code_mermaid = generer_storytelling_mermaid(df_cible)
                except Exception as err:
                    st.error(f"Impossible de générer le storytelling : {err}")
                else:
                    st.success(
                        "Storytelling généré. Copiez le code ou téléchargez le fichier Mermaid."
                    )
                    st.code(code_mermaid, language="markdown")
                    st.download_button(
                        "Télécharger le diagramme Mermaid",
                        data=code_mermaid,
                        file_name="storytelling_mermaid.mmd",
                        mime="text/plain",
                        key="dl_storytelling_mermaid",
                    )
                    if "Aucune scène narrative détectée" in code_mermaid:
                        st.info(
                            "Aucun marqueur narratif dominant n'a été trouvé : le diagramme reste minimal."
                        )

    st.markdown("### Schéma actanciel")
    st.markdown(
        "Le schéma actanciel met en relation les rôles (sujet, objet, adjuvant…) dans un récit politique."\
        "\nIl permet d'identifier qui agit, pour qui et contre quoi au fil du discours."
    )
    st.markdown(
        "Le schéma actanciel, dans une perspective d’analyse narrative, est un modèle qui décrit la structure d’un récit non pas à partir des personnages concrets, mais à partir des rôles qu’ils jouent dans l’action. "
        "Il distingue notamment un sujet (celui qui entreprend une quête), un objet (ce qui est recherché ou à atteindre), un destinateur (ce qui pousse à agir: une instance supérieure, une valeur, une institution, une crise...), un destinataire (ceux pour qui l’action est menée ou qui en reçoivent les effets), des adjuvants (ce qui aide le sujet: alliés, institutions, dispositifs, circonstances favorables), et des opposants (ce qui fait obstacle: adversaires, contraintes, résistances, crises)."
    )

    textes_actanciels: Dict[str, str] = {}
    if texte_source.strip():
        textes_actanciels[libelle_discours_1] = texte_source
    if texte_source_2.strip():
        textes_actanciels[libelle_discours_2] = texte_source_2

    if not textes_actanciels:
        st.info("Aucun discours fourni pour réaliser l'analyse actancielle.")
    else:
        choix_actanciel = st.selectbox(
            "Choisissez le discours à analyser selon les actants",
            options=list(textes_actanciels.keys()),
            key="choix_actanciel",
        )

        if st.button("Analyser les actants", key="btn_actanciel"):
            texte_cible = textes_actanciels.get(choix_actanciel, "")
            df_actants = analyser_actants_texte(texte_cible, DICO_ACTANCIEL)
            tableau = construire_tableau_actanciel(df_actants)
            if tableau.empty:
                st.warning("Aucun marqueur actanciel détecté dans ce discours.")
            else:
                st.success("Analyse actancielle terminée.")
                st.dataframe(tableau, use_container_width=True)
                synthese = synthese_roles_actanciels(df_actants)
                if not synthese.empty:
                    st.bar_chart(
                        data=synthese.set_index("role_actanciel"),
                        use_container_width=True,
                    )

with tab_sentiments:
    render_sentiments_tab(
        texte_source,
        texte_source_2,
        libelle_discours_1,
        libelle_discours_2,
    )

with tab_camembert:
    render_camembert_tab(
        texte_source,
        texte_source_2,
        libelle_discours_1,
        libelle_discours_2,
    )

with tab_toxicite:
    render_toxicite_tab(
        texte_source,
        texte_source_2,
        libelle_discours_1,
        libelle_discours_2,
    )

with tab_zero_shot:
    render_zero_shot_tab(
        texte_source,
        texte_source_2,
        libelle_discours_1,
        libelle_discours_2,
    )

with tab_feel:
    render_feel_tab(
        texte_source,
        texte_source_2,
        libelle_discours_1,
        libelle_discours_2,
    )

with tab_stats_norm:
    render_stats_norm_tab(
        texte_source,
        df_conn,
        df_marq,
        df_memoires,
        df_consq_lex,
        df_causes_lex,
        df_tensions,
        texte_source_2=texte_source_2,
        df_conn_2=df_conn_2,
        df_marqueurs_2=df_marq_2,
        df_memoires_2=df_memoires_2,
        df_consq_lex_2=df_consq_lex_2,
        df_causes_lex_2=df_causes_lex_2,
        df_tensions_2=df_tensions_2,
        heading_discours_1=nom_discours_1,
        heading_discours_2=nom_discours_2,
        couleur_discours_1=couleur_discours_1,
        couleur_discours_2=couleur_discours_2,
    )

with tab_discours:
    st.subheader("Comparaison de deux discours")
    st.markdown(
        "Les analyses affichées ci-dessous reposent sur les détections **Regex** (dictionnaires JSON)."
    )
    if not texte_source.strip() and not texte_source_2.strip():
        st.info("Chargez ou saisissez deux discours pour comparer les détections.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<span style="color:{couleur_discours_1}; font-weight:700;">{libelle_discours_1}</span>',
                unsafe_allow_html=True,
            )
            render_detection_section(
                texte_source,
                detections_1,
                key_prefix="disc1_compare_",
                use_regex_cc=use_regex_cc,
                heading_color=couleur_discours_1,
                dico_connecteurs=DICO_CONNECTEURS,
                dico_marqueurs=DICO_MARQUEURS,
                dico_memoires=DICO_MEMOIRES,
                dico_consq=DICO_CONSQS,
                dico_causes=DICO_CAUSES,
                dico_tensions=DICO_TENSIONS,
            )

        with col_b:
            st.markdown(
                f'<span style="color:{couleur_discours_2}; font-weight:700;">{libelle_discours_2}</span>',
                unsafe_allow_html=True,
            )
            render_detection_section(
                texte_source_2,
                detections_2,
                key_prefix="disc2_compare_",
                use_regex_cc=use_regex_cc,
                heading_color=couleur_discours_2,
                dico_connecteurs=DICO_CONNECTEURS,
                dico_marqueurs=DICO_MARQUEURS,
                dico_memoires=DICO_MEMOIRES,
                dico_consq=DICO_CONSQS,
                dico_causes=DICO_CAUSES,
                dico_tensions=DICO_TENSIONS,
            )
