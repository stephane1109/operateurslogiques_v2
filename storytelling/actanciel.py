"""Outils d'analyse actancielle pour les discours."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ==========================
# Chargement du dictionnaire
# ==========================
def charger_dictionnaire_actanciel(chemin_fichier: Path | str) -> Dict[str, dict]:
    """Charge le dictionnaire actanciel au format JSON."""

    chemin = Path(chemin_fichier)
    with chemin.open("r", encoding="utf-8") as f:  # type: ignore[arg-type]
        return json.load(f)


def _compilation_patterns(dictionnaire: Dict[str, dict]) -> Dict[str, List[Tuple[str, re.Pattern]]]:
    """Compile les expressions régulières du dictionnaire actanciel."""

    motifs_compiles: Dict[str, List[Tuple[str, re.Pattern]]] = {}
    for cle, bloc in dictionnaire.items():
        patterns_regex = bloc.get("patterns_regex", [])
        motifs_compiles[cle] = [
            (expr, re.compile(expr))
            for expr in patterns_regex
            if isinstance(expr, str) and expr.strip()
        ]
    return motifs_compiles


def analyser_actants_texte(
    texte: str, dictionnaire: Dict[str, dict]
) -> pd.DataFrame:
    """Identifie les occurrences actancielles dans un texte."""

    if not texte.strip():
        return pd.DataFrame(columns=["code", "role_actanciel", "description", "extrait", "position"])

    motifs_compiles = _compilation_patterns(dictionnaire)
    enregistrements = []
    for cle, bloc in dictionnaire.items():
        role = bloc.get("role_actanciel", "Inconnu")
        description = bloc.get("description", "")
        for expr, pattern in motifs_compiles.get(cle, []):
            for match in pattern.finditer(texte):
                enregistrements.append(
                    {
                        "code": cle,
                        "role_actanciel": role,
                        "description": description,
                        "extrait": texte[match.start() : match.end()],
                        "position": match.start(),
                        "motif": expr,
                    }
                )
    if not enregistrements:
        return pd.DataFrame(columns=["code", "role_actanciel", "description", "extrait", "position", "motif"])
    df = pd.DataFrame(enregistrements)
    return df.sort_values(by="position").reset_index(drop=True)


def synthese_roles_actanciels(df_actants: pd.DataFrame) -> pd.DataFrame:
    """Agrège les occurrences par rôle actanciel."""

    if df_actants.empty:
        return pd.DataFrame(columns=["role_actanciel", "occurrences"])
    synthese = (
        df_actants.groupby("role_actanciel")
        .size()
        .reset_index(name="occurrences")
        .sort_values(by="occurrences", ascending=False)
    )
    return synthese


def construire_tableau_actanciel(df_actants: pd.DataFrame) -> pd.DataFrame:
    """Prépare un tableau lisible des occurrences actancielles."""

    if df_actants.empty:
        return pd.DataFrame(columns=["Rôle", "Extrait", "Description", "Motif", "Position"])
    return df_actants.rename(
        columns={
            "role_actanciel": "Rôle",
            "extrait": "Extrait",
            "description": "Description",
            "motif": "Motif",
            "position": "Position",
        }
    )[["Rôle", "Extrait", "Description", "Motif", "Position"]]
