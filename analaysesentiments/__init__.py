"""Modules d'analyse de sentiments pour l'application Streamlit."""

from .analysebert import render_camembert_tab
from .zeroshotclassification import render_zero_shot_tab
from .AsentToxic import render_toxicite_tab

__all__ = ["render_camembert_tab", "render_zero_shot_tab", "render_toxicite_tab"]
