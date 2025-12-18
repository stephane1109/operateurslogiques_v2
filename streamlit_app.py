"""Entrée principale pour Streamlit Cloud.

Ce wrapper importe simplement ``main.py`` afin de respecter la
convention de nommage attendue par Streamlit Community Cloud
(``streamlit_app.py``).  Ainsi, la configuration par défaut de la
plateforme repère automatiquement l'application sans qu'il soit
nécessaire d'ajuster manuellement le chemin du fichier principal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# Garantit que le répertoire du projet est présent dans sys.path, même si
# l'environnement d'exécution lance le script depuis un autre dossier.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Charge et exécute main.py, qui contient toute la logique Streamlit.
spec = importlib.util.spec_from_file_location("main", ROOT_DIR / "main.py")
if spec is None or spec.loader is None:  # pragma: no cover - garde-fou
    raise ImportError("Impossible de localiser main.py pour lancer l'application.")
module = importlib.util.module_from_spec(spec)
sys.modules["main"] = module
spec.loader.exec_module(module)
