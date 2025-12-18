"""Helpers pour l'affichage Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def dataframe_safe(
    data: pd.DataFrame | None,
    *,
    empty_message: str | None = "Aucune donnée à afficher.",
    **kwargs,
) -> None:
    """Affiche un DataFrame dans Streamlit en interceptant les erreurs courantes.

    - Ignore silencieusement les valeurs ``None`` ou les DataFrames vides (avec un
      message informatif optionnel).
    - Convertit les structures tabulaires simples en ``pd.DataFrame``.
    - Capture les erreurs d'affichage pour éviter de casser l'app et les remonte
      via ``st.error``.
    """
    if data is None:
        if empty_message:
            st.info(empty_message)
        return

    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    except Exception as err:
        st.error(f"Impossible de convertir les données en tableau : {err}")
        return

    if df.empty:
        if empty_message:
            st.info(empty_message)
        return

    try:
        st.dataframe(df, **kwargs)
    except Exception as err:  # pragma: no cover - affichage Streamlit
        st.error(f"Impossible d'afficher le tableau : {err}")

