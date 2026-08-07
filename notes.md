## Petite explication "jury"

### Pourquoi des rectangles ?

Parce que chaque rectangle représente une zone d'intérêt dans le document.

Exemple :

REFERENCE
┌──────────────────────┐
│  ABC-12345           │
└──────────────────────┘

ou

TOTAL_PRICE
┌──────────────────────┐
│  3 254,90 €          │
└──────────────────────┘

Le modèle apprendra progressivement que ce type de zone correspond à une entité métier.

Question probable du jury : Pourquoi avoir choisi une annotation par régions ?

Réponse : Parce que LayoutLMv3 exploite à la fois le contenu textuel et la disposition spatiale du document. Les annotations définissent les régions contenant les informations métier à apprendre.



### Pourquoi Label Studio ne contient-il pas directement le texte ?

Réponse :

Parce que Label Studio est un outil d'annotation. Il indique quelles régions correspondent aux entités métier. Le texte est obtenu par une étape OCR indépendante. Le dataset final est construit en associant les mots détectés par l'OCR aux régions annotées.

Cette réponse montre que tu comprends bien la séparation des responsabilités entre les différents composants du pipeline.

### Difference entre OCR et LayoutLMV3
L'OCR extrait le contenu du document : les mots et leur position. Label Studio n'extrait pas le texte, il apporte la connaissance métier en indiquant quelles zones correspondent aux informations que l'on souhaite apprendre (fournisseur, numéro de devis, montant, etc.). Le pipeline consiste ensuite à associer les mots détectés par l'OCR aux régions annotées afin de produire un dataset supervisé utilisable pour le fine-tuning de LayoutLMv3.

### Concept Dpcument AI

Document
        │
        ▼
OCR
        │
        ├── mots
        └── positions
                 │
                 ▼
Label Studio
        │
        └── régions métier
                 │
                 ▼
Matching géométrique
                 │
                 ▼
Tokens + labels BIO
                 │
                 ▼
LayoutLMv3


### Pourquoi normaliser les bounding boxes ?

Les coordonnées dépendent de la résolution de l’image utilisée par chaque outil. La normalisation permet de représenter les positions dans un référentiel commun indépendant de la taille de l’image et compatible avec l’entrée spatiale de LayoutLMv3.


Voici un résumé que tu peux intégrer dans tes notes.

---

# Notebook 05 – LayoutLMv3 (Résumé de la séance)

## Objectif de la séance

Passer du **dataset préparé** (OCR + annotations + BIO) à une **entrée complète compatible avec LayoutLMv3**.

---

# 1. Compréhension de LayoutLMv3

## Évolution des versions

**LayoutLM v1**

* Texte
* Bounding Boxes

**LayoutLM v2**

* Ajout de l'image
* Extraction visuelle par CNN

**LayoutLM v3**

* Suppression du CNN
* Utilisation d'un **Vision Transformer (ViT)** basé sur des patches d'image
* Fusion de trois modalités :

  * texte
  * positions
  * image

---

# 2. Pourquoi ne pas utiliser l'OCR intégré ?

Décision volontaire.

Pipeline retenu :

```text
PDF
    ↓
OCR (Tesseract aujourd'hui)
    ↓
Words + Bounding Boxes
    ↓
LayoutLMv3
```

Avantages :

* pipeline modulaire
* changement d'OCR sans modifier le reste
* comparaison scientifique (Tesseract vs Mistral OCR)
* meilleure architecture pour le produit Beluo

---

# 3. Découverte du Processor

Chargement :

```python
LayoutLMv3Processor
```

Le Processor contient :

* Image Processor
* Tokenizer
* Gestion des Bounding Boxes

Important :

```python
apply_ocr=False
```

car notre OCR est déjà effectué.

---

# 4. Tokenisation

Exemple :

```text
002.143/25
```

devient

```text
Ġ00
2
.
143
/
25
```

Conclusion :

Un mot OCR peut devenir plusieurs sous-tokens.

---

# 5. Alignement

Tous les sous-tokens héritent :

* de la même Bounding Box
* du même mot OCR

Seul le premier sous-token conserve le label.

Les suivants reçoivent :

```text
-100
```

Ils sont ignorés pendant le calcul de la loss.

---

# 6. Premier encodage

Création du premier :

```python
encoding = processor(...)
```

Obtention des tenseurs :

* input_ids
* attention_mask
* bbox
* labels
* pixel_values

Dimensions :

```text
input_ids      [1,512]
attention_mask [1,512]
bbox           [1,512,4]
labels         [1,512]
pixel_values   [1,3,224,224]
```

---

# 7. Chargement du modèle

Création :

```python
LayoutLMv3ForTokenClassification
```

avec :

```python
num_labels
label2id
id2label
```

La tête de classification est recréée pour nos labels métier.

Le reste du modèle provient du pré-entraînement.

---

# 8. Premier Forward Pass

Exécution :

```python
outputs = model(**encoding)
```

Résultat :

```text
Loss : 2.45

Logits : [1,512,11]
```

Le pipeline complet fonctionne.

---

# 9. Compréhension des logits

Le modèle ne prédit pas directement un label.

Il produit un score pour chacun des 11 labels.

Exemple :

```text
Sud

↓

[
-1.2,
3.6,
0.4,
...
]
```

Le label retenu est celui ayant le score maximal (`argmax`).

---

# 10. Compréhension de la Loss

Principe :

```text
Prédiction

↓

Comparaison avec la vérité terrain

↓

CrossEntropyLoss

↓

Correction des poids
```

Plus la prédiction est mauvaise, plus la Loss est élevée.

---

# 11. Différence entre train() et eval()

```python
model.train()
```

* mode apprentissage
* Dropout actif

---

```python
model.eval()
```

* mode inférence
* Dropout désactivé

---

```python
torch.no_grad()
```

* aucun calcul de gradients
* moins de mémoire
* plus rapide

---

# 12. Ce qu'il reste

Prochaine séance :

* création du dataset complet avec les 7 devis
* TrainingArguments
* Trainer Hugging Face
* premier fine-tuning
* premières métriques (Loss, Precision, Recall, F1)

---

## État d'avancement

Le pipeline complet est maintenant validé :

```text
PDF
    ↓
OCR
    ↓
Annotations Label Studio
    ↓
BIO
    ↓
Dataset Hugging Face
    ↓
Processor
    ↓
LayoutLMv3
    ↓
Forward Pass
    ↓
Loss + Logits
```

La prochaine étape consiste à passer d'un **exemple unique** à un **entraînement sur l'ensemble des 7 devis**. C'est le passage de la validation technique au véritable apprentissage du modèle.

Oui, exactement. **L'attention mask est une notion fondamentale des Transformers**, pas spécifique à LayoutLMv3.

C'est même l'un des éléments que tu retrouveras dans quasiment tous les modèles de la famille BERT, RoBERTa, CamemBERT, LayoutLM, etc.

---

## Pourquoi existe-t-il ?

Tous les documents n'ont pas la même longueur.

Par exemple :

Document A :

```text
Bonjour Monsieur
```

→ 12 tokens

Document B :

```text
Bonjour Monsieur,

Suite à notre entretien...
...
```

→ 380 tokens

Or, un Transformer traite les données sous forme de tenseurs de taille fixe dans un batch.

Il faut donc que tous les exemples aient la même longueur.

---

## Le padding

Dans LayoutLMv3, on a demandé :

```python
padding="max_length"
max_length=512
```

Supposons que ton document contienne seulement 180 tokens.

Le Processor ajoute alors :

```text
180 vrais tokens

+

332 tokens de padding
```

pour obtenir :

```text
512 tokens
```

---

## Le problème

Le Transformer ne sait pas que ces 332 derniers tokens sont "vides".

Sans information supplémentaire, il pourrait essayer de leur prêter attention.

C'est exactement le rôle de l'**attention mask**.

---

## L'attention mask

Il contient simplement :

```text
1 = vrai token

0 = padding
```

Par exemple :

```text
Tokens

Bonjour
Monsieur
,
...
<pad>
<pad>
<pad>

↓

Attention Mask

1
1
1
...
0
0
0
```

---

## Comment le Transformer l'utilise ?

Le mécanisme d'attention calcule en permanence :

> "À quels autres tokens dois-je prêter attention ?"

Sans masque :

```text
"Bonjour"

↓

peut regarder

Monsieur

,

<pad>

<pad>

<pad>
```

Ce qui n'a aucun sens.

Avec le masque :

```text
Attention Mask

1 1 1 0 0 0
```

Les positions contenant un `0` sont ignorées.

Le Transformer ne calcule jamais d'attention vers ces tokens.

---

## Dans LayoutLMv3

Tu avais :

```python
encoding.keys()
```

qui donnait :

```text
input_ids
attention_mask
bbox
labels
pixel_values
```

Chaque entrée joue un rôle différent :

| Élément        | Rôle                                |
| -------------- | ----------------------------------- |
| input_ids      | Les tokens                          |
| attention_mask | Quels tokens sont réels             |
| bbox           | Où se trouvent les tokens           |
| pixel_values   | L'image du document                 |
| labels         | Les annotations pour l'entraînement |

---

## Ne pas confondre avec les labels à `-100`

C'est une confusion très fréquente.

Les deux servent à **ignorer des éléments**, mais pas au même moment.

### `attention_mask`

Ignore les **tokens de padding**.

```text
PAD PAD PAD

↓

0 0 0
```

Le Transformer ne les lit même pas.

---

### `labels = -100`

Ignore les **sous-tokens** pendant le calcul de la loss.

Par exemple :

```text
Sud-Ouest

↓

Sud
-
O
uest
```

On obtient :

```text
B-FOURNISSEUR
-100
-100
-100
```

Ici, le Transformer traite bien les quatre sous-tokens. En revanche, **seul le premier est utilisé pour calculer la perte**.

---

## Une analogie

Imagine une salle d'examen de 512 places.

Ton document n'occupe que 180 places.

Les 332 autres sont vides.

L'**attention mask** est la liste des sièges occupés :

```text
1 = un étudiant est assis

0 = siège vide
```

L'examinateur ne pose des questions qu'aux étudiants présents.

Ensuite, parmi les étudiants présents, certains représentent un même mot découpé en plusieurs sous-tokens. Pour ces sous-tokens, le `-100` indique simplement : *"ne note pas cette réponse, elle est déjà représentée par le premier sous-token."*

C'est cette combinaison (`attention_mask` pour le Transformer et `labels=-100` pour la fonction de perte) qui permet de gérer efficacement des documents de longueur variable et des mots découpés en plusieurs tokens.

### Learning rate
Plus le modèle est pré-entraîné et plus ton dataset est petit, plus le learning rate doit être faible.


Deux architectures multimodales ont été comparées avec les mêmes données et les mêmes paramètres d’entraînement : LayoutLMv3 et LiLT.

LayoutLMv3 obtient le meilleur score F1 avec 0,727, contre 0,715 pour LiLT. LiLT présente une précision légèrement supérieure, 0,648 contre 0,636, mais LayoutLMv3 obtient un meilleur recall, 0,848 contre 0,798.

Dans le contexte de l’extraction documentaire, le recall est important afin de limiter les champs métier manquants. LayoutLMv3 a donc été retenu pour la phase d’hyperparamétrage.

Les performances des époques 9 et 10 étant presque identiques, les résultats montrent également que l’entraînement a atteint un plateau.

LayoutLMv3 et LiLT ont été évalués sur un jeu de test identique composé de 23 documents et 35 pages. LayoutLMv3 obtient un score F1 de 0,689 contre 0,658 pour LiLT. Il présente également une meilleure précision, 0,591 contre 0,571, et un meilleur rappel, 0,827 contre 0,776.

Bien que LiLT soit plus rapide en inférence, LayoutLMv3 a été retenu car il offre le meilleur compromis entre précision et rappel et généralise mieux sur les documents non vus. Le rappel supérieur est particulièrement important dans ce projet, puisqu’il réduit le risque de champs métier manquants.

Modèle retenu : LayoutLMv3
F1 test       : 0,6894
Motif         : meilleur F1, meilleure précision et meilleur recall
Étape suivante : hyperparamétrage limité du learning rate

Une recherche ciblée du learning rate a été réalisée afin d'améliorer les performances du modèle LayoutLMv3. Trois valeurs étaient initialement envisagées (5e-5, 3e-5 et 2e-5). Après comparaison des deux premières, le learning rate 3e-5 a montré une amélioration simultanée de la précision, du rappel et du score F1 (+1,4 point). Compte tenu des contraintes temporelles du projet et du gain obtenu, cette valeur a été retenue pour le modèle final sans poursuivre l'exploration de 2e-5.

Trois valeurs du learning rate ont été évaluées (5e-5, 3e-5 et 2e-5). Les autres hyperparamètres ont été conservés constants. Le meilleur compromis est obtenu avec 3e-5, qui améliore le score F1 de 0,727 à 0,741 tout en augmentant simultanément la précision et le rappel. Une diminution supplémentaire du learning rate (2e-5) entraîne une baisse des performances, ce qui indique un apprentissage insuffisant.