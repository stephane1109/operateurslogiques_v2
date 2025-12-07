"""Fonctions utilitaires pour l'import/export des dictionnaires JSON.

Ces fonctions préparent les données pour Streamlit (téléchargement) et
valident les fichiers JSON importés par l'utilisateur avant de les
appliquer aux analyses.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Callable


def dictionnaire_to_bytes(dico: Dict[str, Any]) -> bytes:
    """Sérialise un dictionnaire en JSON lisible (UTF-8, indenté)."""
    return json.dumps(dico, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def parse_uploaded_dictionary(
    uploaded_file: Any,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Dict[str, str]:
    """Valide et normalise un dictionnaire JSON importé via Streamlit.

    Les clés vides sont ignorées et les valeurs sont systématiquement
    uppercased pour rester cohérentes avec le reste de l'application.
    Un normaliseur peut être fourni pour homogénéiser les clés
    (espaces, apostrophes…).
    """
    if uploaded_file is None:
        raise ValueError("Aucun fichier fourni.")

    contenu = uploaded_file.getvalue()
    try:
        donnees = json.loads(contenu.decode("utf-8"))
    except Exception as err:  # pragma: no cover - erreurs remontées à Streamlit
        raise ValueError(f"Impossible de lire le JSON : {err}") from err

    if not isinstance(donnees, dict):
        raise ValueError("Le fichier doit contenir un objet JSON {clé: étiquette}.")

    resultat: Dict[str, str] = {}
    for cle, etiquette in donnees.items():
        if cle is None or str(cle).strip() == "":
            continue
        cle_str = str(cle)
        if normalizer is not None:
            cle_str = normalizer(cle_str)
        resultat[cle_str.lower()] = str(etiquette).upper()

    if not resultat:
        raise ValueError("Le JSON ne contient aucune entrée exploitable.")
    return resultat
