{
  "sujet_locuteur_gouvernement": {
    "type": "actant",
    "role_actanciel": "Sujet",
    "description": "Formules où le locuteur et le gouvernement se présentent comme sujets d'action, porteurs de la mission, de la décision ou d'un geste stratégique (proposer, suspendre, réformer).",
    "formules_typiques": [
      "J'ai accepté la mission que m'a confié le président de la République",
      "J'ai proposé un gouvernement de mission",
      "Le gouvernement proposera, nous débattrons, vous voterez",
      "Le gouvernement a une première mission d'urgence, donner un budget sérieux et fiable à la France",
      "Je proposerai un principe simple",
      "Je décide de suspendre la réforme",
      "Suspendre la réforme n'a de sens que si c'est pour aller plus loin"
    ],
    "patterns_regex": [
      "(?i)\\bje (veux|souhaite|m'engage|propose|vous parle|vous dis|m'adresse à vous)\\b",
      "(?i)\\bj'ai (accept[ée] la mission|pris acte|propos[ée] un gouvernement)\\b",
      "(?i)\\b(le|ce) gouvernement (proposera|pr[ée]sentera|a une premi[èe]re mission|incarne|poursuivra)\\b",
      "(?i)\\bmon gouvernement\\b",
      "(?i)\\bce gouvernement (incarne|porte|pr[ée]sente)\\b",
      "(?i)\\ble gouvernement proposera\\b",
      "(?i)\\bnous ferons des propositions\\b",
      "(?i)\\bje d[ée]cide de suspendre\\b",
      "(?i)\\bnous (d[ée]cidons|avons d[ée]cid[ée]) de suspendre\\b",
      "(?i)\\bsuspendre (la|cette) r[ée]forme\\b"
    ]
  },

  "destinataire_parlement_citoyens": {
    "type": "actant",
    "role_actanciel": "Destinataire",
    "description": "Formules qui désignent ceux à qui l'action et le discours sont destinés ou qui en reçoivent les effets: Parlement, sénateurs, députés, Français, concitoyens.",
    "formules_typiques": [
      "Mesdames et messieurs les parlementaires",
      "Mesdames et messieurs les députés",
      "Mesdames et messieurs les sénateurs",
      "Les Français n'attendent pas moins de leurs représentants",
      "Nos concitoyens veulent que le pouvoir soit proche d'eux",
      "Vous voterez",
      "Vous le voterez"
    ],
    "patterns_regex": [
      "(?i)\\bmesdames et messieurs les (parlementaires|d[ée]put[ée]s|s[ée]nateurs?)\\b",
      "(?i)\\bmesdames, messieurs\\b",
      "(?i)\\bmesdames et messieurs\\b",
      "(?i)\\bles Fran[cç]ais(?:es)?\\b",
      "(?i)\\bnos concitoyens\\b",
      "(?i)\\bnos compatriotes\\b",
      "(?i)\\bles Cal[ée]doniens\\b",
      "(?i)\\bles (collectivit[ée]s locales|territoires|outre[- ]mer)\\b",
      "(?i)\\bvous voterez\\b",
      "(?i)\\bvous le voterez\\b"
    ]
  },

  "destinateur_republique_interet_general": {
    "type": "actant",
    "role_actanciel": "Destinateur",
    "description": "Formules qui invoquent la République, l'État, la France, la souveraineté ou l'intérêt général comme principe supérieur qui mande et légitime l'action.",
    "formules_typiques": [
      "J'ai accepté la mission que m'a confié le président de la République",
      "Il y a un impératif de souveraineté qui s'impose à nous tous",
      "Au nom de la stabilité du pays",
      "La République nous oblige",
      "L'intérêt général s'impose à tous"
    ],
    "patterns_regex": [
      "(?i)\\bau nom de (la R[ée]publique|la France|l'int[ée]r[êe]t g[ée]n[ée]ral)\\b",
      "(?i)\\bla (R[ée]publique|France) (nous oblige|exige de nous|a besoin de|doit)\\b",
      "(?i)\\bint[ée]r[êe]t g[ée]n[ée]ral\\b",
      "(?i)\\bimp[ée]ratif de souverainet[ée]\\b",
      "(?i)\\bsouverainet[ée] (budg[ée]taire|[ée]conomique|nationale)\\b",
      "(?i)\\bmod[èe]le (social|de retraite|de d[ée]mocratie sociale)\\b"
    ]
  },

  "objet_budget_reformes_stabilite": {
    "type": "actant",
    "role_actanciel": "Objet",
    "description": "Formules qui désignent l'objet principal de la quête politique: budget, réforme des retraites, stabilité, décentralisation, planification écologique, etc.",
    "formules_typiques": [
      "Donner un budget sérieux et fiable à la France",
      "La priorité absolue du gouvernement, c'est le budget",
      "Suspendre la réforme de 2023 sur les retraites",
      "Un nouveau grand acte de décentralisation",
      "Repenser complètement notre planification écologique et énergétique"
    ],
    "patterns_regex": [
      "(?i)\\b(budget|projet de budget|budget 2025|budget de l[’']?[ée]tat|budget de la s[ée]curit[ée] sociale)\\b",
      "(?i)\\br[ée]forme (des retraites|institutionnelle|de l[’']?[ée]tat)\\b",
      "(?i)\\bstabilit[ée] (du pays|politique|budg[ée]taire)\\b",
      "(?i)\\b(planification|transition) (écologique|[ée]nerg[ée]tique)\\b",
      "(?i)\\bacte de d[ée]centralisation\\b"
    ]
  },

  "adjuvant_parlement_senat_partenaires": {
    "type": "actant",
    "role_actanciel": "Adjuvant",
    "description": "Formules qui présentent le Parlement, le Sénat, les partenaires sociaux, les collectivités comme soutiens ou relais pour accomplir la mission.",
    "formules_typiques": [
      "Le gouvernement proposera, nous débattrons, vous voterez",
      "Je crois en la sagesse du Sénat",
      "Le Sénat y prendra toute sa part en responsabilité",
      "Les partenaires sociaux devront s'emparer de cette question centrale",
      "Les collectivités territoriales doivent être associées",
      "Nous débattrons",
      "Vous en débattrez"
    ],
    "patterns_regex": [
      "(?i)\\b(le )?S[ée]nat (y prendra toute sa part|saura le trouver|nous guidera)\\b",
      "(?i)\\bAssembl[ée] nationale\\b",
      "(?i)\\b(parlementaires?|d[ée]put[ée]s|s[ée]nateurs?) (devront|pourront|prendront|prennent part)\\b",
      "(?i)\\bpartenaires sociaux\\b",
      "(?i)\\borganisations (syndicales|patronales)\\b",
      "(?i)\\bcollectivit[ée]s (locales|territoriales)\\b",
      "(?i)\\bservices publics\\b",
      "(?i)\\bnous d[ée]battrons\\b",
      "(?i)\\bvous en d[ée]battrez\\b"
    ]
  },

  "opposant_crises_instabilite_division": {
    "type": "actant",
    "role_actanciel": "Opposant",
    "description": "Formules qui désignent les crises, l'instabilité, la division, la dette ou la dépendance aux marchés comme forces adverses ou contraintes.",
    "formules_typiques": [
      "Cette crise a des racines",
      "Nous vivons et nous vivrons dans une époque de crise",
      "La division a un coût",
      "La trop forte dépendance à des prêteurs étrangers n'est pas acceptable",
      "L'instabilité aurait coûté 12 milliards depuis la censure"
    ],
    "patterns_regex": [
      "(?i)\\b(crise|crises) (parlementaire[s]?|sociales?|[ée]conomiques?|financi[èe]res?|[ée]cologiques?|climatiques?)\\b",
      "(?i)\\binstabilit[ée] (financi[èe]re|politique)\\b",
      "(?i)\\bdivision(s)? (du pays|politiques?)?\\b",
      "(?i)\\bd[ée]pendance (durable )?à des pr[êe]teurs [ée]trangers\\b",
      "(?i)\\b(co[ûu]t|prix) de l'(instabilit[ée]|incertitude politique)\\b",
      "(?i)\\bd[ée]ficit(s)? (publics?|budg[ée]taires?)\\b"
    ]
  },

  "opposant_blocages_oppositions": {
    "type": "actant",
    "role_actanciel": "Opposant",
    "description": "Formules qui désignent les oppositions, blocages, dogmatismes ou rentes comme forces qui empêchent ou freinent l'action du sujet.",
    "formules_typiques": [
      "Certains questionnent, à juste titre, notre capacité collective à faire de réelles économies",
      "Aucune mesure ne doit être repoussée a priori par dogmatisme",
      "La panne de la construction affecte toute l'économie",
      "Ceux qui ne changent pas, ceux qui s'agrippent aux vieux réflexes disparaîtront"
    ],
    "patterns_regex": [
      "(?i)\\boppositions?\\b",
      "(?i)\\bblocages? (administratifs?|politiques?|sociaux?)\\b",
      "(?i)\\bdogmatisme\\b",
      "(?i)\\beffets de rente\\b",
      "(?i)\\bpanne (de la construction|budg[ée]taire)\\b",
      "(?i)\\bceux qui (refusent|s'agrippent|ne veulent pas|ne changent pas)\\b"
    ]
  },

  "formule_triptyque_proposer_debattre_voter": {
    "type": "formule_narrative",
    "description": "Formule récurrente qui distribue les rôles (gouvernement, Parlement, vote) et scénarise la procédure: proposer / débattre / voter.",
    "formules_typiques": [
      "Le gouvernement proposera, nous débattrons, vous voterez",
      "Nous ferons des propositions, nous débattrons et à la fin, vous voterez",
      "Nous le proposerons, vous en débattrez, vous le voterez"
    ],
    "patterns_regex": [
      "(?i)le gouvernement proposera,? nous d[ée]battrons,? vous voterez",
      "(?i)nous ferons des propositions,? nous d[ée]battrons et (à la fin,? )?vous voterez",
      "(?i)nous le proposerons,? vous en d[ée]battrez,? vous le voterez"
    ]
  },

  "formule_suspendre_pour_mieux_faire": {
    "type": "formule_narrative",
    "description": "Formule qui met en scène la suspension d'une réforme comme moment de crise maîtrisée ouvrant la voie à un nouvel accord.",
    "formules_typiques": [
      "Suspendre pour suspendre n'a aucun sens",
      "Suspendre la réforme n'a de sens que si c'est pour aller plus loin",
      "Suspendre ce n'est pas renoncer, ce n'est pas reculer",
      "La suspension doit installer la confiance nécessaire pour bâtir de nouvelles solutions"
    ],
    "patterns_regex": [
      "(?i)\\bsuspendre (pour suspendre )?n['’]a (aucun|pas de) sens\\b",
      "(?i)\\bsuspendre (la|cette) r[ée]forme\\b",
      "(?i)\\bsuspendre ce n['’]est pas (renoncer|reculer)\\b",
      "(?i)\\bcette suspension (doit|devra) (installer|cr[ée]er) la confiance\\b"
    ]
  }
}
