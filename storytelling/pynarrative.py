"""Fonctions de storytelling pour transformer des phrases annotées en scènes narratives.

Ce module propose une construction basique de scènes à partir des phrases
annotées puis une génération d'un diagramme Mermaid (flowchart TD)
pour visualiser le récit sous forme de scènes enchaînées.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence
import html

import pandas as pd

# Configuration des motifs associés à chaque type de scène.
# Les motifs correspondent à des fragments de noms de colonnes booléennes
# présents dans df_phrases (voir afc.construire_df_phrases).
TYPE_MOTIFS: Dict[str, List[str]] = {
    "diagnostic": ["cause_", "consequence_", "tension_"],
    "cadrage_normatif": ["marqueur_obligation", "marqueur_interdiction", "marqueur_sanction"],
    "legitimation": ["memoire_", "marqueur_legitimation", "marqueur_fonction"],
    "excuse_reparation": ["marqueur_try_catch", "memoire_"],
    "projection_promesse": ["marqueur_permission", "marqueur_recommandation", "consequence_"],
}

# Priorité appliquée en cas d'égalité sur les scores des types narratifs.
ORDRE_PRIORITE_TYPES: Sequence[str] = (
    "diagnostic",
    "cadrage_normatif",
    "legitimation",
    "excuse_reparation",
    "projection_promesse",
)


@dataclass
class SceneNarrative:
    """Représentation d'une scène narrative."""

    id_scene: str
    type_scene: str
    indices_phrases: List[int]
    resume: str = ""
    marqueurs_dominants: List[str] = field(default_factory=list)


# ===========================================================
# Fonctions internes (analyse des marqueurs par bloc de phrases)
# ===========================================================
def _extraire_colonnes(df: pd.DataFrame, motif: str) -> List[str]:
    """Retourne les colonnes contenant le motif indiqué."""

    return [col for col in df.columns if motif in col]


def _compter_occurrences(df_scene: pd.DataFrame, motifs: Iterable[str]) -> Dict[str, int]:
    """Compte les occurrences de chaque motif (somme des booléens correspondants)."""

    compte: Dict[str, int] = {}
    for motif in motifs:
        colonnes = _extraire_colonnes(df_scene, motif)
        if not colonnes:
            compte[motif] = 0
            continue
        compte[motif] = int(df_scene[colonnes].fillna(False).astype(int).sum().sum())
    return compte


def _determiner_type_scene(df_scene: pd.DataFrame) -> tuple[str, List[str]]:
    """Détermine le type de scène dominant et les marqueurs les plus forts."""

    scores_types: Dict[str, int] = {}
    compte_motifs: Dict[str, int] = {}
    for type_scene, motifs in TYPE_MOTIFS.items():
        compte = _compter_occurrences(df_scene, motifs)
        compte_motifs.update(compte)
        scores_types[type_scene] = sum(compte.values())

    max_score = max(scores_types.values()) if scores_types else 0
    if max_score == 0:
        return "narration_generale", []

    types_meilleurs = [t for t, score in scores_types.items() if score == max_score]
    for prioritaire in ORDRE_PRIORITE_TYPES:
        if prioritaire in types_meilleurs:
            type_retenu = prioritaire
            break
    else:  # pragma: no cover - garde-fou
        type_retenu = types_meilleurs[0]

    max_motif = max(compte_motifs.values()) if compte_motifs else 0
    marqueurs_dominants = [m for m, val in compte_motifs.items() if val == max_motif and val > 0]
    return type_retenu, sorted(marqueurs_dominants)


def _resumer_bloc(phrases: Sequence[str], longueur_max: Optional[int] = None) -> str:
    """Construit un résumé à partir d'une liste de phrases sans tronquer par défaut."""

    texte = " ".join([p for p in phrases if isinstance(p, str)]).strip()
    if longueur_max is not None and len(texte) > longueur_max:
        return texte[:longueur_max].rstrip() + "…"
    return texte


def _couleur_motif(motif: str) -> str:
    """Retourne une couleur hexadécimale associée à un motif narratif."""

    palette = {
        "cause_": "#2f855a",
        "consequence_": "#b00020",
        "tension_": "#7b1fa2",
        "marqueur_obligation": "#a86600",
        "marqueur_interdiction": "#c62828",
        "marqueur_sanction": "#ad1457",
        "memoire_": "#1565c0",
        "marqueur_legitimation": "#3f51b5",
        "marqueur_fonction": "#5e35b1",
        "marqueur_permission": "#2e7d32",
        "marqueur_recommandation": "#0d9488",
    }

    for prefixe, couleur in palette.items():
        if motif.startswith(prefixe):
            return couleur
    return "#0b4f6c"


def _styliser_marqueurs(marqueurs: List[str]) -> str:
    """Retourne les marqueurs colorés via des balises span pour Mermaid."""

    marqueurs_colors = []
    for motif in marqueurs:
        couleur = _couleur_motif(motif)
        motif_esc = html.escape(motif, quote=False)
        marqueurs_colors.append(
            f'<span style="color:{couleur}; font-weight:600;">{motif_esc}</span>'
        )
    return ", ".join(marqueurs_colors)


def _decouper_en_blocs(df_phrases: pd.DataFrame) -> List[List[int]]:
    """Crée des blocs de phrases à partir des cadres d'ouverture/fermeture."""

    if df_phrases.empty:
        return []

    col_ouv = next((c for c in df_phrases.columns if "marqueur_cadre_ouverture" in c), None)
    col_ferm = next((c for c in df_phrases.columns if "marqueur_cadre_fermeture" in c), None)

    blocs: List[List[int]] = []
    bloc_courant: List[int] = []

    for idx, row in df_phrases.iterrows():
        if col_ouv and bool(row.get(col_ouv, False)) and bloc_courant:
            blocs.append(bloc_courant)
            bloc_courant = []

        bloc_courant.append(idx)

        if col_ferm and bool(row.get(col_ferm, False)):
            blocs.append(bloc_courant)
            bloc_courant = []

    if bloc_courant:
        blocs.append(bloc_courant)

    return blocs


# ===========================================================
# API publique
# ===========================================================
def construire_scenes_narratives(df_phrases: pd.DataFrame) -> List[SceneNarrative]:
    """Construit une liste ordonnée de scènes narratives à partir de df_phrases."""

    if df_phrases is None or df_phrases.empty:
        return []

    blocs = _decouper_en_blocs(df_phrases)
    if not blocs:
        blocs = [list(df_phrases.index)]

    scenes: List[SceneNarrative] = []
    for i, indices_bloc in enumerate(blocs, start=1):
        sous_df = df_phrases.loc[indices_bloc]
        type_scene, marqueurs_dominants = _determiner_type_scene(sous_df)

        ids_phrases = (
            pd.to_numeric(sous_df.get("id_phrase"), errors="coerce")
            .fillna(pd.Series(range(1, len(sous_df) + 1), index=sous_df.index))
            .astype(int)
            .tolist()
        )
        resume = _resumer_bloc(sous_df.get("texte_phrase", []))
        scenes.append(
            SceneNarrative(
                id_scene=f"S{i}",
                type_scene=type_scene,
                indices_phrases=ids_phrases,
                resume=resume,
                marqueurs_dominants=marqueurs_dominants,
            )
        )

    return scenes


def _format_label_scene(scene: SceneNarrative) -> str:
    """Formate le libellé affiché dans le noeud Mermaid."""

    type_libelle = html.escape(scene.type_scene.replace("_", " ").capitalize(), quote=False)
    details = []
    if scene.marqueurs_dominants:
        marqueurs_colorises = _styliser_marqueurs(scene.marqueurs_dominants)
        details.append(f"Marqueurs : {marqueurs_colorises}")
    if scene.resume:
        details.append(html.escape(scene.resume, quote=False))
    contenu = f"Scène {scene.id_scene[1:]} : {type_libelle}"
    if details:
        contenu += "<br/>" + "<br/>".join(details)
    return contenu.replace('"', "\"")


def generer_mermaid_flowchart(scenes: List[SceneNarrative]) -> str:
    """Génère un flowchart Mermaid linéaire à partir des scènes."""

    lignes = [
        "%%{init: {'flowchart': {'htmlLabels': true}, 'themeVariables': {'fontFamily': 'Inter, Arial, sans-serif'}}}%%",
        "flowchart TD",
    ]
    if not scenes:
        lignes.append('    A["Aucune scène narrative détectée"]')
        return "\n".join(lignes)

    for scene in scenes:
        label = _format_label_scene(scene)
        lignes.append(f'    {scene.id_scene}["{label}"]')

    for courant, suivant in zip(scenes, scenes[1:]):
        lignes.append(f"    {courant.id_scene} --> {suivant.id_scene}")

    return "\n".join(lignes)


def generer_storytelling_mermaid(df_phrases: pd.DataFrame) -> str:
    """Fonction de haut niveau : construit les scènes puis renvoie le code Mermaid."""

    scenes = construire_scenes_narratives(df_phrases)
    return generer_mermaid_flowchart(scenes)
