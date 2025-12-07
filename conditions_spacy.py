"""Analyse des structures conditionnelles (SI / ALORS / SINON) à l'aide de spaCy.

Ce module complète les heuristiques regex présentes dans ``main.py`` en
proposant une extraction basée sur l'analyse dépendancielle offerte par le
modèle ``fr_core_news_md`` (ou, à défaut, ``fr_core_news_sm``).

Fonction principale
-------------------

``analyser_conditions_spacy``
    Retourne un ``DataFrame`` avec les segments condition → conséquence →
    alternative détectés par spaCy. Les colonnes sont :

    - ``id_phrase`` : index (1-based) de la phrase analysée par spaCy.
    - ``declencheur`` : forme de surface du marqueur conditionnel repéré.
    - ``condition`` : sous-arbre considéré comme la condition (protase).
    - ``apodose`` : portion consécutive à la condition (conséquence « alors »).
    - ``alternative`` : portion liée à une alternative (sinon/à défaut…).
    - ``phrase`` : texte complet de la phrase spaCy utilisée.
    - ``methode`` : rappel de la méthode (spaCy / dépendances).

L'objectif n'est pas de remplacer les heuristiques existantes mais d'offrir un
regard complémentaire, surtout lorsque l'on dispose d'un modèle spaCy de taille
moyenne (``fr_core_news_md``) fournissant des dépendances plus fiables.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Pré-traitements sur les déclencheurs
# ---------------------------------------------------------------------------

def _normaliser_terme(terme: str) -> Tuple[str, ...]:
    """Normalise un terme multi-mots en tuple de tokens en minuscules."""

    if not terme:
        return tuple()
    # Remplace les apostrophes typographiques et homogénéise les espaces
    canonique = (
        terme.replace("’", "'").replace("`", "'").replace("\xa0", " ")
    )
    canonique = " ".join(p.strip() for p in canonique.split())
    if not canonique:
        return tuple()
    return tuple(canonique.lower().split())


def _preparer_termes(termes: Iterable[str]) -> List[Tuple[str, ...]]:
    """Filtre et normalise une collection de déclencheurs textuels."""

    motifs: List[Tuple[str, ...]] = []
    for terme in termes:
        motif = _normaliser_terme(terme)
        if motif:
            motifs.append(motif)
    # Trie du plus long au plus court pour éviter les chevauchements prématurés
    motifs.sort(key=len, reverse=True)
    return motifs


def _trouver_locutions(
    tokens: Sequence,
    motifs: Sequence[Tuple[str, ...]],
    start_index: int = 0,
) -> List[Tuple[int, int, Tuple[str, ...]]]:
    """
    Recherche les locutions parmi une séquence de tokens spaCy.

    Parameters
    ----------
    tokens: Sequence[Token]
        Les tokens (d'une phrase) parmi lesquels chercher les motifs.
    motifs: Sequence[Tuple[str, ...]]
        Les motifs normalisés (minuscules) à détecter.
    start_index: int
        Indice minimal à partir duquel effectuer la recherche.

    Returns
    -------
    List[Tuple[int, int, Tuple[str, ...]]]
        Liste de triplets ``(start, end, motif)`` où ``start`` est inclusif et
        ``end`` exclusif (mêmes conventions que spaCy).
    """

    if not motifs:
        return []

    trouvailles: List[Tuple[int, int, Tuple[str, ...]]] = []
    lowers = [tok.lower_ for tok in tokens]

    for motif in motifs:
        longueur = len(motif)
        if longueur == 0:
            continue
        for idx in range(start_index, len(tokens) - longueur + 1):
            segment = lowers[idx : idx + longueur]
            if list(segment) == list(motif):
                trouvailles.append((idx, idx + longueur, motif))

    # Trie chronologique (premières occurrences en priorité)
    trouvailles.sort(key=lambda it: (it[0], -len(it[2])))
    return trouvailles


# ---------------------------------------------------------------------------
# Extraction conditionnelle via dépendances spaCy
# ---------------------------------------------------------------------------

def _texte_depuis_indices(doc, start: int, end: int) -> str:
    """Retourne ``doc[start:end]`` en gérant les bornes invalides."""

    if start is None or end is None or start >= end:
        return ""
    return doc[start:end].text


def _nettoyer_segment(segment: str) -> str:
    """Uniformise les segments extraits pour l'affichage."""

    return segment.strip(" \t\n\r,;:—-·")


def analyser_conditions_spacy(
    texte: str,
    nlp,
    termes_condition: Iterable[str],
    termes_alors: Iterable[str],
    termes_sinon: Iterable[str],
) -> pd.DataFrame:
    """Analyse les structures conditionnelles d'un texte à l'aide de spaCy.

    Parameters
    ----------
    texte: str
        Texte d'entrée à analyser.
    nlp: ``spacy.language.Language``
        Pipeline spaCy déjà chargé.
    termes_condition / termes_alors / termes_sinon: Iterable[str]
        Déclencheurs lexicaux extraits des dictionnaires JSON.

    Returns
    -------
    pd.DataFrame
        Tableau des segments conditionnels. Retourne un ``DataFrame`` vide si
        aucun modèle n'est fourni ou si le texte est vide.
    """

    colonnes = [
        "id_phrase",
        "declencheur",
        "condition",
        "apodose",
        "alternative",
        "phrase",
        "methode",
    ]

    if not texte or not texte.strip() or nlp is None:
        return pd.DataFrame(columns=colonnes)

    motifs_condition = _preparer_termes(termes_condition)
    motifs_alors = _preparer_termes(termes_alors)
    motifs_sinon = _preparer_termes(termes_sinon)

    if not motifs_condition:
        return pd.DataFrame(columns=colonnes)

    doc = nlp(texte)
    phrases = list(doc.sents)

    enregistrements = []

    for idx_phrase, phrase in enumerate(phrases, start=1):
        tokens = list(phrase)
        if not tokens:
            continue

        matches_condition = _trouver_locutions(tokens, motifs_condition)
        if not matches_condition:
            continue

        prochaine_phrase = phrases[idx_phrase] if idx_phrase < len(phrases) else None

        for start, end, motif in matches_condition:
            # Token ancre (= dernier token de la locution)
            anchor_token = tokens[end - 1]
            motif_txt = _texte_depuis_indices(doc, tokens[start].i, anchor_token.i + 1)

            # Sous-arbre de la condition (protase)
            head = anchor_token.head
            if head.sent != phrase:
                head = anchor_token
            subtree_tokens = [tok for tok in head.subtree if tok.sent == phrase]
            if not subtree_tokens:
                subtree_tokens = [anchor_token]

            cond_start = min(tok.i for tok in subtree_tokens)
            cond_end = max(tok.i for tok in subtree_tokens)
            condition_txt = _nettoyer_segment(
                _texte_depuis_indices(doc, cond_start, cond_end + 1)
            )

            # Par défaut, apodose = texte après la condition dans la même phrase
            apres_condition_start = cond_end + 1
            apodose_txt = _nettoyer_segment(
                _texte_depuis_indices(doc, apres_condition_start, phrase.end)
            )

            alternative_txt = ""

            # Recherche d'un connecteur « alors » dans la même phrase
            matches_alors = [
                m
                for m in _trouver_locutions(tokens, motifs_alors, start_index=end)
                if tokens[m[0]].i >= apres_condition_start
            ]
            if matches_alors:
                premier_alors = matches_alors[0]
                start_alors = tokens[premier_alors[0]].i
                end_alors = tokens[premier_alors[1] - 1].i + 1
                apodose_txt = _nettoyer_segment(
                    _texte_depuis_indices(doc, end_alors, phrase.end)
                )
                # Ce qui précède « alors » et suit la condition complète l'apodose
                if start_alors > apres_condition_start:
                    prefixe = _nettoyer_segment(
                        _texte_depuis_indices(
                            doc, apres_condition_start, start_alors
                        )
                    )
                    if prefixe:
                        apodose_txt = (prefixe + " " + apodose_txt).strip()

            # Recherche d'un connecteur « sinon » dans la même phrase
            matches_sinon = [
                m
                for m in _trouver_locutions(tokens, motifs_sinon, start_index=end)
                if tokens[m[0]].i >= apres_condition_start
            ]
            if matches_sinon:
                premier_sinon = matches_sinon[0]
                start_sinon = tokens[premier_sinon[0]].i
                fin_sinon = tokens[premier_sinon[1] - 1].i + 1
                alternative_txt = _nettoyer_segment(
                    _texte_depuis_indices(doc, fin_sinon, phrase.end)
                )
                # Ce qui précède « sinon » nourrit l'apodose (si vide)
                if apodose_txt:
                    apodose_txt = _nettoyer_segment(
                        _texte_depuis_indices(
                            doc, apres_condition_start, start_sinon
                        )
                    )
                else:
                    apodose_txt = _nettoyer_segment(
                        _texte_depuis_indices(
                            doc, apres_condition_start, start_sinon
                        )
                    )

            # Fallback : si aucune apodose détectée, regarder la phrase suivante
            if not apodose_txt and prochaine_phrase is not None:
                tokens_next = list(prochaine_phrase)
                matches_alors_next = _trouver_locutions(tokens_next, motifs_alors, 0)
                matches_sinon_next = _trouver_locutions(tokens_next, motifs_sinon, 0)

                if matches_alors_next:
                    premier = matches_alors_next[0]
                    fin = (
                        tokens_next[premier[1]].i
                        if premier[1] < len(tokens_next)
                        else prochaine_phrase.end
                    )
                    apodose_txt = _nettoyer_segment(
                        _texte_depuis_indices(doc, fin, prochaine_phrase.end)
                    )
                elif tokens_next:
                    apodose_txt = _nettoyer_segment(
                        _texte_depuis_indices(doc, prochaine_phrase.start, prochaine_phrase.end)
                    )

                if matches_sinon_next:
                    premier = matches_sinon_next[0]
                    fin = (
                        tokens_next[premier[1]].i
                        if premier[1] < len(tokens_next)
                        else prochaine_phrase.end
                    )
                    alternative_txt = _nettoyer_segment(
                        _texte_depuis_indices(doc, fin, prochaine_phrase.end)
                    )

            enregistrements.append(
                {
                    "id_phrase": idx_phrase,
                    "declencheur": motif_txt,
                    "condition": condition_txt,
                    "apodose": apodose_txt,
                    "alternative": alternative_txt,
                    "phrase": phrase.text.strip(),
                    "methode": "spaCy (dépendances)",
                }
            )

    if not enregistrements:
        return pd.DataFrame(columns=colonnes)

    return pd.DataFrame(enregistrements, columns=colonnes)


__all__ = ["analyser_conditions_spacy"]

