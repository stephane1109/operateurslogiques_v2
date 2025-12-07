"""Helpers de compatibilité pour Streamlit."""

from __future__ import annotations

import inspect
from typing import Any

import streamlit as st


def dataframe_safe(data: Any, **kwargs):
    """Affiche un dataframe en ignorant les arguments non pris en charge.

    Certains déploiements peuvent utiliser une version de Streamlit ne
    comprenant pas encore le paramètre ``hide_index``. Cette fonction retire
    l'argument si nécessaire et réessaie en cas d'erreur de type.
    """

    params = inspect.signature(st.dataframe).parameters
    if "hide_index" not in params:
        kwargs.pop("hide_index", None)

    try:
        return st.dataframe(data, **kwargs)
    except TypeError:
        kwargs.pop("hide_index", None)
        return st.dataframe(data, **kwargs)
