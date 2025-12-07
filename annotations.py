"""Onglet d'annotation interactive pour générer des dictionnaires JSON.

Ce module propose une interface Streamlit permettant à l'utilisateur de :
- définir ses propres labels (marqueurs) d'annotation ;
- sélectionner des segments de texte à annoter via des bornes de positions ;
- visualiser les surlignages directement dans l'application ;
- exporter les annotations au format JSON.

L'interface s'inspire de la librairie "streamlit-annotation-tools" tout en
restant autonome (pas d'installation supplémentaire requise).
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
from typing import Dict, List

import streamlit as st


_COLOR_PALETTE = [
    "#fde68a",
    "#bbf7d0",
    "#bfdbfe",
    "#fecdd3",
    "#ddd6fe",
    "#fbcfe8",
    "#fed7aa",
    "#c7d2fe",
]


@dataclass
class Annotation:
    label: str
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, str | int]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@st.cache_data(show_spinner=False)
def _default_labels() -> List[str]:
    return ["Personne", "Organisation", "Lieu"]


def _color_for_label(label: str, existing: Dict[str, str]) -> str:
    if label in existing:
        return existing[label]
    color = _COLOR_PALETTE[len(existing) % len(_COLOR_PALETTE)]
    existing[label] = color
    return color


def _render_highlighted_text(text: str, annotations: List[Annotation]) -> None:
    if not text.strip():
        st.info("Ajoutez d'abord un texte à annoter.")
        return

    if not annotations:
        st.write(text)
        return

    annotations_sorted = sorted(annotations, key=lambda ann: ann.start)
    cursor = 0
    label_colors: Dict[str, str] = {}
    html_parts: List[str] = []
    plain_text = st.session_state.get("annotation_plain_text", text)

    for ann in annotations_sorted:
        if ann.start > cursor:
            html_parts.append(html.escape(plain_text[cursor:ann.start]))
        color = _color_for_label(ann.label, label_colors)
        highlighted = (
            f"<span style='background:{color}; padding:2px 4px; border-radius:4px;'>"
            f"{html.escape(ann.text)}</span>"
        )
        html_parts.append(highlighted)
        cursor = ann.end

    if cursor < len(text):
        html_parts.append(html.escape(text[cursor:]))

    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _sync_label_checkboxes(labels: List[str]) -> None:
    """Nettoie les cases à cocher des labels supprimés et initialise les nouveaux."""
    prefix = "annotation_label_"
    current_keys = {key for key in st.session_state if key.startswith(prefix)}
    valid_keys = {f"{prefix}{label}" for label in labels}

    # Supprime les anciennes cases devenues obsolètes
    for stale in current_keys - valid_keys:
        st.session_state.pop(stale, None)

    # Initialise explicitement les nouvelles cases à False si absentes
    for new_key in valid_keys - current_keys:
        st.session_state[new_key] = False


def render_annotation_tab(texte_source: str) -> None:
    st.subheader("Annotation manuelle du texte")
    st.caption(
        "Définissez vos marqueurs, cochez le label souhaité, sélectionnez une portion du texte"
        " puis ajoutez l'annotation. Les annotations sont exportables au format JSON."
    )

    st.session_state.setdefault("annotation_labels", _default_labels())
    st.session_state.setdefault("annotations", [])

    texte = texte_source or ""
    previous_text = st.session_state.get("annotation_plain_text")
    if previous_text is not None and previous_text != texte:
        st.session_state["annotations"] = []

    st.session_state["annotation_plain_text"] = texte

    st.markdown("**Texte importé depuis la section 'Source du discours'**")
    if texte.strip():
        st.markdown(
            "<div style='white-space:pre-wrap; border:1px solid #e0e0e0; padding:0.75rem; "
            "border-radius:0.5rem; background:#fafafa; color:#111;'>" + html.escape(texte) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Aucun texte importé pour le moment.")

    with st.expander("Marqueurs disponibles", expanded=True):
        st.write(
            "Ajoutez ou supprimez des labels. Ils seront proposés comme cases à cocher pour l'annotation."
        )
        nouveau_label = st.text_input("Ajouter un label", placeholder="Ex : Concept", key="label_input")
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Ajouter le label", key="ajout_label"):
                if nouveau_label.strip() and nouveau_label not in st.session_state["annotation_labels"]:
                    st.session_state["annotation_labels"].append(nouveau_label.strip())
                elif nouveau_label.strip():
                    st.warning("Ce label existe déjà.")
        with cols[1]:
            if st.button("Réinitialiser les labels", key="reset_labels"):
                st.session_state["annotation_labels"] = _default_labels()
        if st.session_state["annotation_labels"]:
            st.write("Labels actuels :")
            st.write(", ".join(st.session_state["annotation_labels"]))
        else:
            st.info("Aucun label défini pour le moment.")

    _sync_label_checkboxes(st.session_state["annotation_labels"])

    if texte.strip():
        labels_actuels = st.session_state["annotation_labels"]
        selection_checkbox = []
        st.markdown("**Choisissez le label à appliquer :**")
        for label in labels_actuels:
            key = f"annotation_label_{label}"
            selection_checkbox.append((label, st.checkbox(label, key=key)))

        labels_selectionnes = [lab for lab, checked in selection_checkbox if checked]

        selection = st.text_area(
            "Passage à annoter",
            value="",
            height=120,
            placeholder="Copiez/collez ici le passage exact à annoter (il doit exister dans le texte importé)",
            key="annotation_selection",
        )

        selection_stripped = selection.strip()
        plain_text = st.session_state["annotation_plain_text"]
        position = plain_text.find(selection_stripped) if selection_stripped else -1

        if selection_stripped and position == -1:
            st.warning(
                "Passage non trouvé dans le texte importé. Assurez-vous de copier/coller le segment exact."
            )
        elif selection_stripped:
            st.markdown(f"**Aperçu du surlignage** : {selection_stripped}")
            st.caption("Seule la première occurrence du passage sera annotée.")
        else:
            st.markdown("**Aperçu du surlignage** : (aucune sélection)")

        if st.button("Ajouter l'annotation", key="ajouter_annotation"):
            if len(labels_selectionnes) == 0:
                st.error("Sélectionnez d'abord un label via les cases à cocher.")
            elif len(labels_selectionnes) > 1:
                st.error("Ne cochez qu'un seul label à la fois pour l'annotation.")
            elif not selection_stripped:
                st.error(
                    "Sélectionnez un passage non vide en le copiant/collant depuis le texte importé."
                )
            elif position == -1:
                st.error(
                    "Passage introuvable dans le texte importé : vérifiez la casse, les espaces et la ponctuation."
                )
            else:
                end = position + len(selection_stripped)
                st.session_state["annotations"].append(
                    Annotation(
                        label=labels_selectionnes[0],
                        start=position,
                        end=end,
                        text=plain_text[position:end],
                    )
                )
                st.success(
                    f"Annotation ajoutée avec le label '{labels_selectionnes[0]}'."
                )

    annotations: List[Annotation] = st.session_state.get("annotations", [])
    if annotations:
        st.markdown("### Aperçu des surlignages")
        _render_highlighted_text(texte, annotations)

        st.markdown("### Tableau des annotations")
        df_data = [ann.to_dict() for ann in annotations]
        st.dataframe(df_data, use_container_width=True)

        json_payload = json.dumps(df_data, ensure_ascii=False, indent=2)
        st.download_button(
            "Télécharger les annotations (JSON)",
            data=json_payload.encode("utf-8"),
            file_name="annotations.json",
            mime="application/json",
            key="dl_annotations_json",
        )
    else:
        st.info("Aucune annotation enregistrée pour le moment.")
