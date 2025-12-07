from textwrap import dedent

import streamlit as st


def render_lexique_tab() -> None:
    """Affiche le contenu de l'onglet Lexique."""

    st.markdown("#### Intro")
    st.markdown(
        dedent(
            """Le script repose sur des dictionnaires au format JSON et sur un ensemble de règles regex,
            qui servent à repérer précisément des motifs linguistiques définis à l’avance.
            Les regex sont moins souples qu’une approche par NLP automatique comme SpaCy,
            mais elles permettent, par leur caractère très ciblé, d’atteindre un niveau de précision et de contrôle dans
            l’analyse. Même si je travaille actuellement sur une comparaison systématique entre les règles regex et le modèle NLP de SpaCy,
            l’intérêt des regex est de garantir une grande precision dans l'analyse."""
        )
    )

    st.markdown("#### Connecteurs logique (façon python!)")
    st.markdown(
        dedent(
            """Ce script réalise une analyse automatique de discours en s’appuyant sur la logique conditionnelle de la
            programmation (notamment Python). Il parcourt le corpus puis applique une série de règles de type « SI… ALORS…
            SINON… TANT QUE » afin de catégoriser le texte selon ces logiques discursives."""
        )
    )

    st.markdown(
        "- **IF (si…)** : introduit une condition ; ce qui suit dépend du fait que la condition soit vraie.\n"
        "- **ELSE (sinon)** : propose l’alternative quand la condition précédente n’est pas remplie.\n"
        "- **WHILE (tant que)** : action répétée tant qu’une condition reste vraie, marque une persistance.\n"
    )

    st.markdown("#### Marqueurs normatifs")
    st.markdown(
        dedent(
            """Le dictionnaire « Marqueurs normatifs » regroupe l’ensemble des formes linguistiques et expressions récurrentes qui
            signalent, dans le discours (politique), des prises de position normatives ou des cadrages spécifiques. Le script
            identifie automatiquement, dans le texte, les moments où l’orateur formule une obligation, une interdiction, une
            permission, une recommandation, une sanction, ou encore ouvre ou clôt un cadre discursif."""
        )
    )
    st.markdown(
        "- **OBLIGATION** : ce qui doit être fait (nécessité, devoir).\n"
        "- **INTERDICTION** : ce qui ne doit pas être fait (empêchement, prohibition).\n"
        "- **PERMISSION** : ce qui est autorisé ou possible.\n"
        "- **RECOMMANDATION** : ce qu’il vaut mieux faire (conseil, souhaitable).\n"
        "- **SANCTION** : annonce une punition ou un coût en cas d’écart.\n"
        "- **CADRE_OUVERTURE** : invite à débattre ou à élargir le dialogue.\n"
        "- **CADRE_FERMETURE** : clôt ou restreint le débat (pas le moment, pas de polémique)."
    )
    st.markdown("#### Marqueurs \"cause/conséquence\"")
    st.markdown(
        "Les dictionnaires « Causes » et « Conséquences ». Ces dictionnaires rassemblent les expressions, connecteurs et constructions syntaxiques qui signalent dans le discours des relations de causalité (ce qui produit, motive ou explique) et des relations de conséquence (ce qui résulte, découle ou est présenté comme l’effet d’une cause)."
    )

    st.markdown(
        "- **CAUSE** : justifie ou explique un fait (parce que, car, en raison de…).\n"
        "- **CONSEQUENCE** : en déduit l’effet ou l’issue (donc, alors, par conséquent…)."
    )

    st.markdown("#### Marqueurs \"mémoire\" (cognitifs)")
    st.markdown(
        "Les marqueurs de « mémoire » (cognitifs) désignent l’ensemble des formes linguistiques par lesquelles le discours mobilise, organise ou oriente le rapport au passé, au souvenir et à l’expérience collective. Dans le cadre d’un discours politique, ces marqueurs signalent les moments où l’orateur convoque des événements antérieurs, des références historiques, des expériences partagées ou des promesses déjà formulées, afin de structurer la compréhension du présent et d’orienter les attentes futures."
    )
    st.markdown(
        "- **MEM_PERS** : souvenirs ou expériences personnelles mobilisés dans le discours.\n"
        "- **MEM_COLL** : mémoire partagée, appel au « nous » collectif.\n"
        "- **MEM_RAPPEL** : injonctions ou formules pour ne pas oublier un fait.\n"
        "- **MEM_RENVOI** : renvoi à des propos ou engagements déjà formulés.\n"
        "- **MEM_REPET** : marqueurs de répétition ou d’insistance (encore, déjà…).\n"
        "- **MEM_PASSE** : ancrage explicite dans un temps passé (jadis, auparavant…)."
    )

    st.markdown("#### Marqueurs Tensions sémantiques")
    st.markdown(
        "Les « tensions sémantiques » renvoient aux moments du discours où les significations sont ambivalentes. Dans un discours politique, ces tensions apparaissent lorsqu’un même segment associe des termes, des cadres ou des valeurs difficilement conciliables," 
    )

    st.markdown(
        "- **CAUSE** : justifie ou explique un fait (parce que, car, en raison de…).\n"
        "- **CONSEQUENCE** : en déduit l’effet ou l’issue (donc, alors, par conséquent…)."
    )

    st.markdown("#### Lexique grammatical spaCy (causalité)")
    st.markdown(
        "- **CAUSE_GN** : cause exprimée par un groupe nominal (souvent introduit par *en raison de*, *à cause de*…). Exemple : *En raison de la tempête, le match est reporté.*\n"
        "- **CONSEQUENCE_ADV** : conséquence exprimée par un adverbe ou un groupe adverbial (ex. *donc*, *par conséquent*). Exemple : *Il a tout expliqué, par conséquent nous comprenons la décision.*\n"
        "- **CAUSE_SUBORDONNEE** : cause formulée par une proposition subordonnée (ex. *parce que*, *puisque*…). Exemple : *Parce qu’il pleuvait, la cérémonie a été déplacée à l’intérieur.*"
    )
