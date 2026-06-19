# Pistes d’amélioration pour le simulateur

## Intégration des capacités de stockage
- Prendre en compte les capacités de stockage des acheteurs et des vendeurs afin de simuler des comportements de spéculation.
  - Exemple : un agent peut stocker un coproduit lorsque les prix sont bas en prévision d’une hausse future.

## Extension de la variable de coût de transport (Cij)
- Inclure dans la variable Cij les coûts de transformation nécessaires pour rendre le coproduit utilisable :
  - Logistique
  - Traitement
  - Conformité réglementaire

## Répartition du risque d’investissement
- Déterminer qui supporte le risque d’investissement dans les infrastructures (ex. transformation) :
  - Le vendeur
  - L’acheteur
  - Un tiers (ex. subventions publiques)

## Temporalité individualisée des agents
- Permettre à chaque agent d’avoir sa propre dynamique temporelle :
  - Fréquence de décision
  - Réactivité
  - Horizon de planification
- Prendre en compte la taille et/ou la matière utilisée :
  - Ex. saisonnalité des bioressources vs. absence de saisonnalité des ressources fossiles

## Marché apprenant
- Intégrer une capacité d’apprentissage dans le marché ou chez les agents pour simuler l’évolution des comportements.
- Intégrer l’entrée/sortie d’acteurs :
  - Nouveaux entrants attirés par l’économie circulaire
  - Sortie des acteurs non satisfaits

## Quantification du gaspillage
- Calculer la quantité de coproduit gaspillée (non achetée ou non utilisée) pour évaluer l’efficacité du système.

## Analyse de la puissance des acteurs
- Évaluer la puissance d’un acteur par :
  - Valeur de Shapley
  - Centralité dans le réseau d’échanges
- Impacts :
  - Un acteur central peut fixer les prix
  - Influence sur la trajectoire technologique du système

## Prise en compte de facteurs non-économiques
- Intégrer des critères non monétaires :
  - Éthique
  - Durabilité
  - Impact environnemental
- Influence de l’usage du coproduit sur le prix :
  - Ex. prix différent selon que le coproduit est vendu à un méthaniseur ou un industriel cosmétique
- Intégrer la capacité d’interaction des acteurs :
  - Confiance partagée
  - Connaissances communes
  - Réseau interpersonnel évolutif

## Marché à ressources finies
- Simuler un marché où la ressource s’épuise progressivement :
  - Hausse ou oscillation des prix liée à la rareté

## Validation empirique du modèle
- Confronter les résultats du simulateur avec :
  - Données réelles
  - Études de cas

## Manipulation écologique des prix
- Étudier l’impact de mécanismes de régulation :
  - Subventions
  - Taxes
  - Priorités d’achat
- Objectif : favoriser les comportements écologiques

---

## Autres pistes techniques

- **Adaptive Buyer Parameters** : Faire évoluer beta selon le succès des achats ou la rareté du marché.
- **Market Shock Testing** : Simuler des chocs de prix ou de rareté en cours de simulation.
- **Modélisation de l’utilité marginale** : Diminution de la volonté de payer à mesure que le besoin (q_needed) diminue.
- **Influence des taxes sur les produits entrants** :
  - Étudier l’impact sur la quantité échangée
  - Remarque : la notion de coût de production n’est pas encore prise en compte
- **Analyse contrefactuelle** :
  - Comparer la récompense obtenue par un agent avec la meilleure possible
  - En testant toutes les bins avec gel de l’apprentissage
