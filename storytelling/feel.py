"""Analyse psycholinguistique basée sur le lexique FEEL.

Ce module propose une granularité émotionnelle fine (joie, colère, peur,
tristesse, surprise, dégoût) pour l'analyse de discours. Il peut exploiter
un fichier CSV du lexique FEEL placé dans ``dictionnaires/feel.csv``
ou s'appuyer sur un échantillon minimal inclus pour des démonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import re
from typing import Dict, Iterable, List, Optional, Set

import altair as alt
import pandas as pd
import streamlit as st

from text_utils import normaliser_espace, segmenter_en_phrases

# Lexique minimal intégré pour fonctionner même sans fichier externe.
# Les colonnes du CSV attendu :
# - format long : "lemme" (ou "word"), "emotion", "polarity".
# - format large FEEL : "id", "word", "polarity", "joy", "fear",
#   "sadness", "anger", "surprise", "disgust" (séparateur ";").
LEXIQUE_FEEL_REDUIT = [
    {"lemme": "joie", "emotion": "joy", "polarity": "positive"},
    {"lemme": "bonheur", "emotion": "joy", "polarity": "positive"},
    {"lemme": "heureux", "emotion": "joy", "polarity": "positive"},
    {"lemme": "colère", "emotion": "anger", "polarity": "negative"},
    {"lemme": "rage", "emotion": "anger", "polarity": "negative"},
    {"lemme": "haine", "emotion": "anger", "polarity": "negative"},
    {"lemme": "peur", "emotion": "fear", "polarity": "negative"},
    {"lemme": "crainte", "emotion": "fear", "polarity": "negative"},
    {"lemme": "angoisse", "emotion": "fear", "polarity": "negative"},
    {"lemme": "tristesse", "emotion": "sadness", "polarity": "negative"},
    {"lemme": "malheur", "emotion": "sadness", "polarity": "negative"},
    {"lemme": "pleurer", "emotion": "sadness", "polarity": "negative"},
    {"lemme": "surprise", "emotion": "surprise", "polarity": "positive"},
    {"lemme": "étonnement", "emotion": "surprise", "polarity": "positive"},
    {"lemme": "imprévu", "emotion": "surprise", "polarity": "positive"},
    {"lemme": "dégoût", "emotion": "disgust", "polarity": "negative"},
    {"lemme": "écœurement", "emotion": "disgust", "polarity": "negative"},
    {"lemme": "répugnance", "emotion": "disgust", "polarity": "negative"},
]


@dataclass
class EmotionScore:
    """Score agrégé pour une émotion donnée."""

    emotion: str
    polarite: str
    occurrences: int
    poids: float
    discours: str


class FeelLexiqueErreur(RuntimeError):
    """Erreur liée au chargement ou au format du lexique FEEL."""


_DEF_CHEMIN_FEEL = Path(__file__).resolve().parent.parent / "dictionnaires" / "feel.csv"
POLARITE_NON_SPECIFIEE = "non spécifiée"


EMOTION_COLORS: Dict[str, str] = {
    "joy": "#f6c445",
    "anger": "#e15759",
    "fear": "#8c6bb1",
    "sadness": "#4e79a7",
    "surprise": "#edc948",
    "disgust": "#59a14f",
}


def _normaliser_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Uniformise les colonnes attendues du lexique FEEL.

    Le CSV peut être livré soit en format « long » (colonnes ``lemme``,
    ``emotion``, ``polarite``), soit en format « large » utilisé par le
    lexique FEEL officiel avec une colonne par émotion. Dans ce dernier cas,
    la fonction déplie les émotions présentes (valeur non nulle) et conserve
    la polarité associée.
    """

    df = df.copy()
    colonnes_equivalentes: Dict[str, str] = {
        "word": "lemme",
        "term": "lemme",
        "polarity": "polarite",
        "valence": "polarite",
    }
    df.rename(columns=colonnes_equivalentes, inplace=True)

    colonnes_emotions = ["joy", "fear", "sadness", "anger", "surprise", "disgust"]
    colonnes_large = set(colonnes_emotions).issubset(df.columns)

    if "emotion" not in df.columns and colonnes_large:
        if "lemme" not in df.columns:
            raise FeelLexiqueErreur(
                "Le lexique FEEL doit contenir une colonne 'lemme' ou 'word'."
            )
        if "polarite" not in df.columns:
            raise FeelLexiqueErreur(
                "Le lexique FEEL doit contenir une colonne 'polarity' ou 'polarite'."
            )

        df = (
            df.melt(
                id_vars=["lemme", "polarite"],
                value_vars=colonnes_emotions,
                var_name="emotion",
                value_name="score",
            )
            .loc[lambda d: d["score"].fillna(0) != 0]
            .drop(columns=["score"])
        )
    else:
        colonnes_manquantes = {"lemme", "emotion", "polarite"} - set(df.columns)
        if colonnes_manquantes:
            raise FeelLexiqueErreur(
                "Le lexique FEEL doit contenir les colonnes 'lemme', 'emotion' et 'polarite'."
            )

        df = df[["lemme", "emotion", "polarite"]]

    df["lemme"] = df["lemme"].astype(str).str.lower().str.strip()
    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
    df["polarite"] = df["polarite"].astype(str).str.lower().str.strip()
    return df[["lemme", "emotion", "polarite"]]


def charger_lexique_feel(chemin: Optional[Path | str] = None) -> pd.DataFrame:
    """Charge le lexique FEEL depuis un CSV ou l'échantillon minimal.

    Parameters
    ----------
    chemin:
        Emplacement du CSV FEEL. Si ``None`` et si le fichier ``dictionnaires/feel.csv``
        existe, il sera utilisé. Sinon un petit lexique intégré sera retourné.
    """

    if chemin is None:
        chemin = _DEF_CHEMIN_FEEL

    if chemin and Path(chemin).exists():
        try:
            df = pd.read_csv(chemin)
            if df.shape[1] == 1 and df.columns[0].count(";") >= 2:
                df = pd.read_csv(chemin, sep=";")
        except Exception as err:  # pragma: no cover - dépendance externe
            raise FeelLexiqueErreur(
                f"Impossible de charger le lexique FEEL depuis {chemin} : {err}"
            ) from err
        return _normaliser_colonnes(df)

    return _normaliser_colonnes(pd.DataFrame(LEXIQUE_FEEL_REDUIT))


def _tokeniser_texte(texte: str) -> List[str]:
    """Découpe le texte en tokens alphanumériques et gère les accents."""

    if not texte:
        return []
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ'-]+", texte.lower())
    return tokens


def _indexer_lexique(lexique: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    """Prépare un index {lemme: [{emotion, polarite}, ...]} pour des accès rapides."""

    if lexique.empty:
        return {}
    lexique_dedup = lexique.drop_duplicates(subset=["lemme", "emotion", "polarite"])
    return {
        lemme: lignes[["emotion", "polarite"]].to_dict("records")
        for lemme, lignes in lexique_dedup.groupby("lemme")
    }


def analyser_emotions_feel(
    texte: str, lexique: pd.DataFrame, discours: str, consider_polarites: bool = True
) -> List[EmotionScore]:
    """Calcule la distribution des émotions FEEL pour un discours."""

    texte_norm = normaliser_espace(texte)
    tokens = _tokeniser_texte(texte_norm)
    if not tokens or lexique.empty:
        return []

    df_tokens = pd.DataFrame({"lemme": tokens})
    df_tokens["lemme"] = df_tokens["lemme"].astype(str).str.lower()
    jointure = df_tokens.merge(lexique, on="lemme", how="inner")
    if jointure.empty:
        return []

    tot_tokens = len(tokens)
    group_cols = ["emotion", "polarite"] if consider_polarites else ["emotion"]
    agregat = (
        jointure.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "occurrences"})
    )
    if not consider_polarites:
        agregat["polarite"] = POLARITE_NON_SPECIFIEE
    agregat["poids"] = agregat["occurrences"] / float(tot_tokens)
    return [
        EmotionScore(
            emotion=row["emotion"],
            polarite=row["polarite"],
            occurrences=int(row["occurrences"]),
            poids=float(row["poids"]),
            discours=discours,
        )
        for _, row in agregat.iterrows()
    ]


def _scores_en_dataframe(scores: Iterable[EmotionScore]) -> pd.DataFrame:
    """Transforme les scores en DataFrame."""

    data = [s.__dict__ for s in scores]
    if not data:
        return pd.DataFrame(
            columns=["emotion", "polarite", "occurrences", "poids", "discours"]
        )
    return pd.DataFrame(data)


def _etiquette_emotion_polarite(entree: Dict[str, str], consider_polarites: bool) -> str:
    """Formate une étiquette lisible pour une entrée du lexique."""

    if not consider_polarites:
        return entree["emotion"]
    return f"{entree['emotion']} ({entree['polarite']})"


def _construire_html_texte_annotes(
    texte: str,
    lexique: pd.DataFrame,
    etiquettes_selectionnees: Optional[Set[str]] = None,
    consider_polarites: bool = True,
) -> str:
    """Construit le HTML du texte original avec les lexèmes annotés FEEL."""

    if not texte.strip() or lexique.empty:
        return ""

    index_lexique = _indexer_lexique(lexique)
    if not index_lexique:
        return ""

    fragments: List[str] = []
    last_end = 0
    for match in re.finditer(r"[\wÀ-ÖØ-öø-ÿ'-]+", texte):
        start, end = match.span()
        fragments.append(html.escape(texte[last_end:start]))
        lemme = match.group(0).lower()
        entrees = index_lexique.get(lemme, [])
        etiquettes = {
            _etiquette_emotion_polarite(e, consider_polarites) for e in entrees
        }
        etiquettes_a_afficher = (
            etiquettes if not etiquettes_selectionnees else etiquettes & etiquettes_selectionnees
        )
        if etiquettes_a_afficher:
            entrees_filtrees = [
                e
                for e in entrees
                if _etiquette_emotion_polarite(e, consider_polarites)
                in etiquettes_a_afficher
            ]
            emotion_principale = entrees_filtrees[0]["emotion"]
            couleur = EMOTION_COLORS.get(emotion_principale, "#bbbbbb")
            fragments.append(
                "<span style=\"background-color:{couleur}; padding:2px 6px; border-radius:6px;\">"
                "{mot}<small style=\"opacity:0.8;\">({etiquettes})</small></span>".format(
                    couleur=couleur,
                    mot=html.escape(match.group(0)),
                    etiquettes=html.escape(", ".join(sorted(etiquettes_a_afficher))),
                )
            )
        else:
            fragments.append(html.escape(match.group(0)))
        last_end = end

    fragments.append(html.escape(texte[last_end:]))
    return "".join(fragments)


def _annoter_texte_avec_emotions(
    texte: str,
    lexique: pd.DataFrame,
    etiquettes_selectionnees: Optional[Set[str]] = None,
    consider_polarites: bool = True,
):
    """Affiche le texte original en surlignant les lexèmes annotés FEEL."""

    html_annotes = _construire_html_texte_annotes(
        texte,
        lexique,
        etiquettes_selectionnees=etiquettes_selectionnees,
        consider_polarites=consider_polarites,
    )
    if not html_annotes:
        st.info("Aucun texte ou lexique FEEL pour afficher des étiquettes.")
        return html_annotes

    st.markdown(html_annotes, unsafe_allow_html=True)
    return html_annotes


def _proportions_temporelles(
    texte: str, lexique: pd.DataFrame, discours: str, consider_polarites: bool = True
) -> pd.DataFrame:
    """Calcule la proportion des émotions/polarités par phrase."""

    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm) if texte_norm else []
    if not phrases or lexique.empty:
        return pd.DataFrame(
            columns=["id_phrase", "emotion", "polarite", "proportion", "occurrences", "discours"]
        )

    index_lexique = _indexer_lexique(lexique)
    data: List[Dict[str, object]] = []

    for idx, phrase in enumerate(phrases, start=1):
        tokens = _tokeniser_texte(phrase)
        mentions: List[Dict[str, str]] = []
        for token in tokens:
            mentions.extend(index_lexique.get(token, []))

        total_mentions = len(mentions)
        if total_mentions == 0:
            continue

        compteur: Dict[tuple[str, str], int] = {}
        for entree in mentions:
            cle = (
                entree["emotion"],
                entree["polarite"] if consider_polarites else POLARITE_NON_SPECIFIEE,
            )
            compteur[cle] = compteur.get(cle, 0) + 1

        for (emotion, polarite), occ in compteur.items():
            data.append(
                {
                    "id_phrase": idx,
                    "emotion": emotion,
                    "polarite": polarite,
                    "proportion": occ / float(total_mentions),
                    "occurrences": occ,
                    "discours": discours,
                }
            )

    return pd.DataFrame(data)


def _visualiser_proportions(df: pd.DataFrame, titre: str):
    """Affiche l'évolution temporelle des émotions/polarités."""

    if df.empty:
        st.info("Aucune émotion FEEL détectée pour construire le graphique temporel.")
        return

    chart = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("id_phrase:Q", title="Temps (phrases)"),
            y=alt.Y(
                "proportion:Q",
                stack="normalize",
                title="Proportion normalisée des émotions détectées",
            ),
            color=alt.Color("emotion:N", title="Émotion"),
            tooltip=[
                "id_phrase",
                "emotion",
                "polarite",
                "occurrences",
                alt.Tooltip("proportion:Q", format=".2%"),
            ],
        )
        .properties(title=titre)
    )
    st.altair_chart(chart, use_container_width=True)


def _afficher_intro_methodologie():
    """Texte introductif sur le lexique FEEL."""

    st.markdown("### FEEL — French Expanded Emotion Lexicon")
    st.markdown(
        "Contrairement aux Transformers (BERT, CamemBERT) qui \"lisent\" et \"comprennent\""
        " le sens d'une phrase grâce notamment à une \"fenêtre d'attention\", le lexique FEEL"
        " fonctionne sur un principe beaucoup plus basique qu'on appelle le \"Bag of Words\""
        " (Sac de mots)."
    )
    st.markdown(
        "FEEL n'utilise aucun \"modèle\" d'IA, c'est une ressource statique. C'est un fichier"
        " csv contenant 14 000 mots qui sont catégorisés selon une émotion (joie, colère,"
        " peur, tristesse, surprise, dégoût) associés à une polarité positive/négative."
    )
    st.markdown(
        "Le script calcule la fréquence des lexèmes présents dans le texte et leur poids"
        " relatif par rapport au nombre total de mots."
    )
    st.markdown("Référence du modèle : Le Lexique FEEL - fr")


def _visualiser_scores(df: pd.DataFrame, titre: str, consider_polarites: bool):
    """Affiche un histogramme des émotions détectées."""

    if df.empty:
        st.info("Aucune émotion FEEL détectée dans ce discours.")
        return

    color_field = "polarite" if consider_polarites else "emotion"
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("emotion:N", sort="-x", title="Émotion"),
            x=alt.X("occurrences:Q", title="Occurrences"),
            color=alt.Color(f"{color_field}:N", title="Polarité" if consider_polarites else "Émotion"),
            tooltip=["emotion", "polarite", "occurrences", alt.Tooltip("poids:Q", format=".2%")],
        )
        .properties(title=titre)
    )
    st.altair_chart(chart, use_container_width=True)


def render_feel_tab(
    texte_discours_1: str,
    texte_discours_2: str,
    nom_discours_1: str,
    nom_discours_2: str,
):
    """Rendu Streamlit pour l'onglet FEEL."""

    _afficher_intro_methodologie()

    st.markdown("#### Options d'affichage")
    consider_polarites = st.checkbox(
        "Prendre en compte la polarité positive/négative",
        value=True,
        help="Affiche les résultats en distinguant les polarités des émotions.",
        key="feel_consider_polarites",
    )
    emotions_uniquement = st.checkbox(
        "Afficher uniquement les émotions (sans polarités)",
        value=False,
        help="Regroupe les scores par émotion sans distinguer la valence.",
        key="feel_emotions_uniquement",
    )
    if emotions_uniquement:
        consider_polarites = False
    elif not consider_polarites:
        emotions_uniquement = False

    if not texte_discours_1.strip() and not texte_discours_2.strip():
        st.info(
            "Aucun discours fourni. Ajoutez du texte pour lancer l'analyse psycholinguistique FEEL."
        )
        return

    try:
        lexique_feel = charger_lexique_feel()
    except FeelLexiqueErreur as err:
        st.error(str(err))
        return

    discours_disponibles: Dict[str, str] = {}
    if texte_discours_1.strip():
        discours_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        discours_disponibles[nom_discours_2] = texte_discours_2

    onglets = st.tabs(list(discours_disponibles.keys()))
    for tab, (nom, contenu) in zip(onglets, discours_disponibles.items()):
        with tab:
            st.markdown(f"#### {nom}")
            scores = analyser_emotions_feel(
                contenu, lexique_feel, nom, consider_polarites=consider_polarites
            )
            df_scores = _scores_en_dataframe(scores)
            if df_scores.empty:
                st.info("Aucune correspondance FEEL trouvée dans ce discours.")
                continue

            st.dataframe(
                df_scores.sort_values("occurrences", ascending=False),
                use_container_width=True,
            )
            _visualiser_scores(
                df_scores,
                titre=f"Distribution émotionnelle — {nom}",
                consider_polarites=consider_polarites,
            )

            st.markdown("##### Texte annoté (étiquettes émotion/polarité)")
            if consider_polarites:
                etiquettes_disponibles = sorted(
                    {
                        f"{emotion} ({polarite})"
                        for emotion, polarite in lexique_feel[["emotion", "polarite"]]
                        .dropna()
                        .drop_duplicates()
                        .itertuples(index=False)
                    }
                )
            else:
                etiquettes_disponibles = sorted(
                    lexique_feel["emotion"].dropna().drop_duplicates().tolist()
                )
            selection_etiquettes = set(
                st.multiselect(
                    "Sélectionnez les étiquettes à surligner (laissez vide pour tout afficher)",
                    etiquettes_disponibles,
                    key=f"selection_etiquettes_{nom}",
                )
            )
            html_annotes = _annoter_texte_avec_emotions(
                contenu,
                lexique_feel,
                etiquettes_selectionnees=selection_etiquettes,
                consider_polarites=consider_polarites,
            )
            if html_annotes:
                st.download_button(
                    label="Télécharger le texte annoté (HTML)",
                    data=html_annotes.encode("utf-8"),
                    file_name=f"{nom}_feel.html",
                    mime="text/html",
                )

            st.markdown("##### Évolution temporelle des émotions/polarités")
            df_proportions = _proportions_temporelles(
                contenu, lexique_feel, nom, consider_polarites=consider_polarites
            )
            if consider_polarites:
                etiquettes_proportions = sorted(
                    {
                        f"{emotion} ({polarite})"
                        for emotion, polarite in df_proportions[["emotion", "polarite"]]
                        .dropna()
                        .drop_duplicates()
                        .itertuples(index=False)
                    }
                )
            else:
                etiquettes_proportions = sorted(
                    df_proportions["emotion"].dropna().drop_duplicates().tolist()
                )
            selection_proportions = set(
                st.multiselect(
                    "Choix des marqueurs/polarités à afficher",
                    options=etiquettes_proportions,
                    default=etiquettes_proportions,
                    help="Filtrer le streamgraph par combinaison émotion/polarité.",
                    key=f"selection_proportions_{nom}",
                )
            )
            df_proportions_filtres = (
                df_proportions[
                    df_proportions.apply(
                        lambda row: (
                            f"{row['emotion']} ({row['polarite']})"
                            if consider_polarites
                            else row["emotion"]
                        )
                        in selection_proportions,
                        axis=1,
                    )
                ]
                if selection_proportions
                else df_proportions
            )
            if df_proportions_filtres.empty:
                st.info(
                    "Aucune émotion FEEL à afficher pour les marqueurs sélectionnés."
                )
            else:
                _visualiser_proportions(
                    df_proportions_filtres,
                    titre=f"Émotions FEEL au fil du discours — {nom}",
                )
