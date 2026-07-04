# Journal des décisions - Document Intelligence Engine

## Objectif

Construire un moteur capable d'extraire automatiquement les informations métier de devis photovoltaïques.

Première cible :
- référence
- unité
- quantité
- prix de vente unitaire
- prix de vente

---

# Décision 001 — Utiliser des devis réels

Décision

Le benchmark sera réalisé sur des devis fournis par Beluo.

Pourquoi

Les benchmarks publics ne représentent pas les documents réellement utilisés par le client.

Statut

✅ Validé

---

# Décision 002 — Commencer par une baseline

Décision

Ne pas entraîner de modèle Deep Learning immédiatement.

Construire d'abord une baseline robuste.

Pipeline retenu :

PDF
↓
Docling
↓
Markdown
↓
Parser Python
↓
JSON Beluo

Pourquoi

Mesurer ce qui fonctionne avant d'ajouter de la complexité.

Statut

✅ Validé

---

# Décision 003 — Benchmark Docling

Corpus

6 devis réels.

Résultat

- 5 devis correctement extraits.
- 1 devis présentant des difficultés de structure.

Conclusion

Docling constitue une bonne baseline.

Statut

✅ Validé

---

# Décision 004 — Benchmark Mistral OCR

Résultat

Mistral améliore certains cas difficiles.

En revanche, Docling reste plus régulier sur plusieurs devis.

Conclusion

Aucun avantage suffisant pour remplacer Docling.

Statut

✅ Validé

---

# Décision 005 — Pipeline V1

Pipeline retenu

PDF
↓
Docling
↓
Extraction du tableau Markdown
↓
Parser
↓
JSON Beluo

Les autres moteurs restent des solutions de secours.

Statut

✅ En cours