"""Analyse de co-occurrences à partir du texte fourni.

Ce module propose un rendu Streamlit avec tableau, graphique Altair et nuage de mots
pour visualiser les co-occurrences calculées à l'échelle de la phrase ou du document.
"""
from __future__ import annotations

import re
import html
from collections import Counter
from functools import lru_cache
from itertools import combinations
from typing import Iterable, List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_utils import dataframe_safe

try:  # pragma: no cover - dépendance optionnelle à l'import
    from wordcloud import WordCloud
except ImportError:  # pragma: no cover - WordCloud non installée
    WordCloud = None

try:  # pragma: no cover - dépendance optionnelle à l'import
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - spaCy non installé
    spacy = None
    Language = None

try:  # pragma: no cover - spaCy non installé
    from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except ImportError:  # pragma: no cover - spaCy non installé
    SPACY_STOP_WORDS = set()


_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


_SPACY_MODELE_NOM: Optional[str] = None
_SPACY_MODELES_TENTES: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _charger_modele_spacy() -> Optional["Language"]:
    """Charge un modèle spaCy français en le mettant en cache."""

    global _SPACY_MODELE_NOM, _SPACY_MODELES_TENTES

    if spacy is None:
        _SPACY_MODELE_NOM = None
        _SPACY_MODELES_TENTES = ()
        return None

    essais: list[str] = []
    for nom_modele in ("fr_core_news_md", "fr_core_news_sm"):
        essais.append(nom_modele)
        try:
            modele = spacy.load(nom_modele)
        except OSError:
            continue
        except Exception:
            continue
        else:
            _SPACY_MODELE_NOM = nom_modele
            _SPACY_MODELES_TENTES = tuple(essais)
            return modele

    _SPACY_MODELE_NOM = None
    _SPACY_MODELES_TENTES = tuple(essais)
    return None


def _segmenter_en_phrases(texte: str) -> List[str]:
    """Segmente le texte en phrases en se basant sur la ponctuation forte."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]


def _formater_noms_modeles(noms: Iterable[str]) -> str:
    noms_list = [f"'{nom}'" for nom in noms]
    if not noms_list:
        return ""
    if len(noms_list) == 1:
        return noms_list[0]
    return ", ".join(noms_list[:-1]) + f" et {noms_list[-1]}"


def _extraire_mots(
    phrase: str,
    *,
    longueur_min: int = 2,
    modele_spacy: Optional["Language"] = None,
    filtrer_stopwords: bool = True,
) -> List[str]:
    """Extrait des mots en minuscule, filtrés sur la longueur minimale.

    Le paramètre ``filtrer_stopwords`` permet de contrôler la suppression des mots
    outils lorsqu'un modèle spaCy est disponible ou, à défaut, via la liste de
    stopwords fournie par ``spacy.lang.fr``.
    """
    if not phrase:
        return []

    if modele_spacy is None:
        modele_spacy = _charger_modele_spacy()

    if modele_spacy is not None:
        doc = modele_spacy(phrase)
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            if len(token.text) < longueur_min:
                continue
            if filtrer_stopwords and token.is_stop:
                continue
            lemme = token.lemma_.lower() if token.lemma_ else token.text.lower()
            tokens.append(lemme)
        if tokens:
            return tokens

    mots = [m.lower() for m in _WORD_PATTERN.findall(phrase)]
    if filtrer_stopwords:
        stopwords = SPACY_STOP_WORDS if SPACY_STOP_WORDS else set()
    else:
        stopwords = set()
    return [m for m in mots if len(m) >= longueur_min and m not in stopwords]


def _mettre_en_evidence_mots(
    phrase: str,
    mots_cibles: Iterable[str],
    modele_spacy: Optional["Language"] = None,
) -> str:
    """Retourne la phrase en HTML avec les mots cibles entourés de <mark>."""

    mots_cibles_set = {m.lower() for m in mots_cibles if m}
    if not phrase or not mots_cibles_set:
        return html.escape(phrase)

    if modele_spacy is not None:
        try:
            doc = modele_spacy(phrase)
        except Exception:
            doc = None
    else:
        doc = None

    if doc is not None:
        morceaux: list[str] = []
        dernier_index = 0
        for token in doc:
            debut, fin = token.idx, token.idx + len(token.text)
            if debut >= len(phrase):
                continue
            morceaux.append(html.escape(phrase[dernier_index:debut]))
            lemme = token.lemma_.lower() if token.lemma_ else token.text.lower()
            texte_token = html.escape(token.text)
            if lemme in mots_cibles_set:
                morceaux.append(f"<mark>{texte_token}</mark>")
            else:
                morceaux.append(texte_token)
            dernier_index = fin
        morceaux.append(html.escape(phrase[dernier_index:]))
        return "".join(morceaux)

    morceaux: list[str] = []
    dernier_index = 0
    for match in _WORD_PATTERN.finditer(phrase):
        debut, fin = match.start(), match.end()
        morceaux.append(html.escape(phrase[dernier_index:debut]))
        mot = match.group(0)
        if mot.lower() in mots_cibles_set:
            morceaux.append(f"<mark>{html.escape(mot)}</mark>")
        else:
            morceaux.append(html.escape(mot))
        dernier_index = fin
    morceaux.append(html.escape(phrase[dernier_index:]))
    return "".join(morceaux)


def _generer_cooccurrences(
    texte: str,
    *,
    longueur_min: int = 2,
    modele_spacy: Optional["Language"] = None,
    granularite: str = "phrase",
    filtrer_stopwords: bool = True,
) -> Counter[str]:
    """Compte les co-occurrences selon la granularité demandée."""

    compteurs: Counter[str] = Counter()

    if granularite == "document":
        tokens = _extraire_mots(
            texte,
            longueur_min=longueur_min,
            modele_spacy=modele_spacy,
            filtrer_stopwords=filtrer_stopwords,
        )
        if len(tokens) < 2:
            return compteurs
        frequences = Counter(tokens)
        mots_uniques = sorted(frequences.keys())
        for idx, mot1 in enumerate(mots_uniques):
            for mot2 in mots_uniques[idx + 1 :]:
                pair = f"{mot1}_{mot2}"
                compteurs[pair] = frequences[mot1] * frequences[mot2]
        return compteurs

    phrases = _segmenter_en_phrases(texte)
    for phrase in phrases:
        tokens = sorted(
            set(
                _extraire_mots(
                    phrase,
                    longueur_min=longueur_min,
                    modele_spacy=modele_spacy,
                    filtrer_stopwords=filtrer_stopwords,
                )
            )
        )
        if len(tokens) < 2:
            continue
        for mot1, mot2 in combinations(tokens, 2):
            pair = f"{mot1}_{mot2}"
            compteurs[pair] += 1
    return compteurs


def calculer_table_cooccurrences(
    texte: str,
    *,
    longueur_min: int = 2,
    granularite: str = "phrase",
    filtrer_stopwords: bool = True,
) -> pd.DataFrame:
    """Retourne un DataFrame des co-occurrences triées par fréquence décroissante."""
    modele_spacy = _charger_modele_spacy()
    compteur = _generer_cooccurrences(
        texte,
        longueur_min=longueur_min,
        modele_spacy=modele_spacy,
        granularite=granularite,
        filtrer_stopwords=filtrer_stopwords,
    )
    if not compteur:
        return pd.DataFrame(columns=["mot1", "mot2", "pair", "occurrences"])

    donnees = []
    for pair, freq in compteur.most_common():
        mot1, mot2 = pair.split("_", 1)
        donnees.append({
            "mot1": mot1,
            "mot2": mot2,
            "pair": pair,
            "occurrences": freq,
        })

    df = pd.DataFrame(donnees)
    df["occurrences"] = pd.to_numeric(df["occurrences"], errors="coerce").fillna(0).astype(int)
    return df


def _graphique_barres_cooccurrences(df: pd.DataFrame, top_n: int) -> alt.Chart | None:
    """Construit un graphique à barres Altair pour les co-occurrences les plus fréquentes."""
    if df.empty:
        return None

    top_df = df.head(top_n).copy()
    hauteur = max(300, 18 * len(top_df))
    chart = (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X("occurrences:Q", title="Occurrences"),
            y=alt.Y("pair:N", sort="-x", title="Paire de mots"),
            color=alt.Color("occurrences:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[
                alt.Tooltip("mot1:N", title="Mot 1"),
                alt.Tooltip("mot2:N", title="Mot 2"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .properties(height=hauteur)
    )
    return chart


def _nuage_de_mots(df: pd.DataFrame, max_mots: int):
    """Construit un nuage de mots via la librairie WordCloud."""

    if WordCloud is None:
        return None

    if df.empty:
        return None

    data = df.head(max_mots).copy()
    if data.empty:
        return None

    frequences = {
        str(ligne.pair): int(ligne.occurrences)
        for ligne in data.itertuples(index=False)
        if getattr(ligne, "occurrences", 0)
    }

    if not frequences:
        return None

    nuage = WordCloud(
        width=800,
        height=400,
        background_color="white",
        prefer_horizontal=1.0,
        colormap="plasma",
    ).generate_from_frequencies(frequences)

    return nuage.to_array()


def _filtrer_cooccurrences_par_mot_cle(df: pd.DataFrame, mot_cle: str) -> pd.DataFrame:
    """Retourne les co-occurrences qui impliquent le mot-clé fourni."""

    mot_cle_normalise = mot_cle.strip().lower()
    if not mot_cle_normalise or df.empty:
        return pd.DataFrame(columns=[*df.columns, "mot_cle", "mot_associe"])

    masque_mot1 = df["mot1"].str.lower() == mot_cle_normalise
    masque_mot2 = df["mot2"].str.lower() == mot_cle_normalise
    masque = masque_mot1 | masque_mot2

    df_filtre = df.loc[masque].copy()
    if df_filtre.empty:
        return df_filtre

    df_filtre["mot_cle"] = mot_cle_normalise
    df_filtre["mot_associe"] = [
        ligne.mot2 if ligne.mot1.lower() == mot_cle_normalise else ligne.mot1
        for ligne in df_filtre.itertuples(index=False)
    ]
    return df_filtre


def render_cooccurrences_tab(texte_source: str) -> None:
    """Affiche l'onglet Streamlit consacré aux co-occurrences par mot-clé."""
    st.subheader("Analyse des co-occurrences par mot-clé")

    if not texte_source or not texte_source.strip():
        st.info("Saisissez ou chargez un texte pour analyser les co-occurrences.")
        return

    longueur_min = st.slider(
        "Longueur minimale des mots (en caractères)",
        min_value=1,
        max_value=6,
        value=2,
        help="Les mots plus courts que cette valeur sont ignorés pour stabiliser les co-occurrences.",
    )

    st.caption(
        "Les co-occurrences sont calculées phrase par phrase. Deux mots sont considérés "
        "comme co-occurrents s'ils apparaissent dans la même phrase."
    )

    filtrer_stopwords = st.checkbox(
        "Filtrer les stopwords",
        value=True,
        help="Décochez pour conserver tous les mots, y compris les articles et prépositions.",
    )

    modele_spacy = _charger_modele_spacy()
    if filtrer_stopwords and modele_spacy is None:
        if spacy is None:
            st.warning(
                "spaCy n'est pas disponible. Les stopwords ne pourront pas être filtrés."
            )
        else:
            noms_modeles = _SPACY_MODELES_TENTES or ("fr_core_news_md", "fr_core_news_sm")
            st.warning(
                "Les modèles spaCy "
                f"{_formater_noms_modeles(noms_modeles)} n'ont pas pu être chargés. "
                "Le filtrage des stopwords sera limité."
            )
    elif _SPACY_MODELE_NOM:
        message = f"Modèle spaCy chargé : {_SPACY_MODELE_NOM}"
        if _SPACY_MODELE_NOM == "fr_core_news_sm":
            message += " (installez 'fr_core_news_md' pour des résultats enrichis)"
        st.caption(message)
    if not filtrer_stopwords:
        st.caption("Le filtrage des stopwords est désactivé pour cette analyse.")

    mot_cle_saisi = st.text_input(
        "Mot-clé pour une analyse ciblée",
        help="Analyse les co-occurrences limitées au mot indiqué.",
    )
    mot_cle_analyse = mot_cle_saisi.strip()
    if not mot_cle_analyse:
        st.info("Saisissez un mot-clé pour lancer l'analyse ciblée des co-occurrences.")
        return

    df_cooc = calculer_table_cooccurrences(
        texte_source,
        longueur_min=longueur_min,
        granularite="phrase",
        filtrer_stopwords=filtrer_stopwords,
    )

    if df_cooc.empty:
        st.info("Aucune co-occurrence n'a été détectée avec les paramètres actuels.")
        return

    df_mot_cle = _filtrer_cooccurrences_par_mot_cle(df_cooc, mot_cle_analyse)
    if df_mot_cle.empty:
        st.info(
            "Aucune co-occurrence ne contient le mot-clé « "
            f"{html.escape(mot_cle_analyse)} » avec les paramètres sélectionnés."
        )
        return

    df_filtre = df_mot_cle.copy()

    st.markdown(
        "### Co-occurrences associées à « "
        f"{html.escape(mot_cle_analyse)} »"
    )
    dataframe_safe(
        df_filtre[["mot_associe", "pair", "occurrences"]]
        .rename(columns={"mot_associe": "Mot associé"}),
        use_container_width=True,
        hide_index=True,
    )
    total_occurrences = int(df_filtre["occurrences"].sum())
    nb_associes = int(df_filtre["mot_associe"].nunique())
    st.caption(
        f"Le mot-clé apparaît dans {nb_associes} co-occurrence(s) distincte(s) "
        f"pour un total de {total_occurrences} occurrence(s)."
    )

    st.markdown("### Visualisation des co-occurrences")

    max_barres = max(1, min(30, len(df_filtre)))
    min_barres = 1 if max_barres < 3 else 3
    valeur_defaut_barres = min(10, max_barres)
    top_n = st.slider(
        "Nombre de co-occurrences à afficher (barres)",
        min_value=min_barres,
        max_value=max_barres,
        value=valeur_defaut_barres,
    )
    chart_barres = _graphique_barres_cooccurrences(df_filtre, top_n)
    if chart_barres is not None:
        st.altair_chart(chart_barres, use_container_width=True)
    else:
        st.caption("Pas de graphique disponible pour les paramètres sélectionnés.")

    st.markdown("### Nuage de mots des co-occurrences")

    max_nuage = max(1, min(50, len(df_filtre)))
    min_nuage = 1 if max_nuage < 3 else 3
    valeur_defaut_nuage = min(20, max_nuage)
    max_mots = st.slider(
        "Nombre de co-occurrences dans le nuage",
        min_value=min_nuage,
        max_value=max_nuage,
        value=valeur_defaut_nuage,
    )
    image_nuage = _nuage_de_mots(df_filtre, max_mots)
    if image_nuage is not None:
        st.image(image_nuage, use_column_width=True)
    else:
        if WordCloud is None:
            st.caption(
                "Le nuage de mots nécessite l'installation de la librairie WordCloud."
            )
        else:
            st.caption("Le nuage de mots n'a pas pu être généré.")

    st.markdown("### Co-occurrences dans le texte")
    phrases = _segmenter_en_phrases(texte_source)
    if not phrases:
        st.info("Impossible de segmenter le texte en phrases pour afficher les co-occurrences.")
        return

    paires_filtrees = [
        (str(ligne.mot1), str(ligne.mot2))
        for ligne in df_filtre.itertuples(index=False)
    ]
    paires_uniques = list(dict.fromkeys(paires_filtrees))

    cooccurrences_trouvees = False
    for phrase in phrases:
        tokens_phrase = _extraire_mots(
            phrase,
            longueur_min=longueur_min,
            modele_spacy=modele_spacy,
            filtrer_stopwords=filtrer_stopwords,
        )
        if not tokens_phrase:
            continue
        tokens_set = set(tokens_phrase)
        paires_dans_phrase = [
            pair for pair in paires_uniques if pair[0] in tokens_set and pair[1] in tokens_set
        ]
        if not paires_dans_phrase:
            continue

        cooccurrences_trouvees = True
        mots_a_surligner = {mot for paire in paires_dans_phrase for mot in paire}
        phrase_html = _mettre_en_evidence_mots(
            phrase,
            mots_a_surligner,
            modele_spacy=modele_spacy,
        )
        st.markdown(
            f"<div style='margin-bottom:0.25rem'>{phrase_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Co-occurrences : "
            + ", ".join(f"{mot1} – {mot2}" for mot1, mot2 in paires_dans_phrase)
        )

    if not cooccurrences_trouvees:
        st.info(
            "Aucune des co-occurrences conservées ne figure explicitement dans les phrases."
        )
