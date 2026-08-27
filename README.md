# Document Intelligence Engine

Projet de  Document AI pour l'extraction automatique d'informations métier à partir de devis du secteur photovoltaïque.

## Contexte et objectif

Cette étude a été réalisée dans le cadre du projet de fin d’études de la formation Deep Learning d’Alyra, à partir d’un cas d’usage proposé par Beluo.

L'objectif est de transformer automatiquement des devis PDF provenant de différents fournisseurs en données métier structurées.

Cinq informations sont extraites :

- client
- fournisseur / émetteur du devis
- numéro de devis
- date du devis
- montant total

La sortie est retournée au format JSON afin d'être exploitée par une application métier.

---

## Cycle du projet Deep Learning

Le projet couvre les principales étapes d'un projet Deep Learning de bout en bout :

1. collecte et compréhension des données ;
2. benchmark et sélection du moteur OCR ;
3. annotation des données ;
4. préparation et construction du dataset ;
5. séparation train / validation / test ;
6. sélection de modèles pré-entraînés ;
7. fine-tuning ;
8. évaluation et comparaison des modèles ;
9. sélection du modèle final ;
10. intégration et déploiement.

---

## Pipeline Document AI

```text
PDF
 ↓
Conversion en images
 ↓
OCR
 ↓
Tokens + Bounding Boxes
 ↓
Token Classification
 ↓
Agrégation des entités
 ↓
JSON métier
```

---

## Dataset

Le dataset final est constitué de **150 devis photovoltaïques** provenant de plusieurs fournisseurs et présentant des structures et mises en page différentes.

Le split est réalisé au niveau document :

| Ensemble | Documents | Part |
|---|---:|---:|
| Train | 105 | 70 % |
| Validation | 22 | 15 % |
| Test | 23 | 15 % |

Le jeu de test contient **35 pages**.

### Entités annotées

Cinq entités métier sont annotées manuellement avec **Label Studio** :

- `CLIENT`
- `DATE_DEVIS`
- `EMETTEUR_DEVIS`
- `MONTANT_TOTAL`
- `NUMERO_DEVIS`

Les annotations sont converties au format **BIO** pour la tâche de token classification, soit 11 classes incluant la classe `O`.

### Préparation

L'OCR fournit les mots et leurs coordonnées spatiales (*bounding boxes*).

Les annotations Label Studio sont associées aux tokens OCR par correspondance géométrique afin de construire le dataset supervisé.

```text
PDF → Image → OCR → Tokens + Bounding Boxes
                         +
                  Annotations Label Studio
                         ↓
                    Labels BIO
                         ↓
                      Dataset
```

---

## Benchmark OCR

Un benchmark a été réalisé afin de sélectionner le moteur OCR utilisé pour la construction du dataset.

Les moteurs évalués sont :

- Tesseract
- EasyOCR
- PaddleOCR

Les critères étudiés incluent la qualité de reconnaissance, le temps de traitement et l'intégration dans le pipeline Document AI.

**Tesseract** a été retenu pour la suite du projet.

---

## Modèles

Deux architectures pré-entraînées de Document AI ont été fine-tunées sur le même dataset.

### LayoutLMv3

Checkpoint initial :

`microsoft/layoutlmv3-base`

LayoutLMv3 exploite conjointement :

- le texte ;
- la position spatiale des tokens ;
- les informations visuelles de la page.

### LiLT

Checkpoint initial :

`SCUT-DLVCLab/lilt-roberta-en-base`

LiLT exploite les informations textuelles et spatiales du document.

L'utilisation de modèles pré-entraînés permet de bénéficier de représentations déjà apprises sur de grands corpus avant de les spécialiser sur les devis photovoltaïques.

---

## Fine-tuning

Les modèles sont fine-tunés pour une tâche supervisée de **token classification**.

Configuration principale :

| Paramètre | Valeur |
|---|---:|
| Batch size | 2 |
| Weight decay | 0.01 |
| Learning rate LiLT | 5e-5 |
| Learning rate LayoutLMv3 | configurable |
| Epochs | configurable |

Le jeu de validation est utilisé pendant l'entraînement, tandis que le jeu de test reste indépendant jusqu'à l'évaluation finale.

---

## Évaluation

Les modèles sont évalués sur un jeu de test indépendant de **23 documents / 35 pages**.

Les principales métriques utilisées sont :

- Precision
- Recall
- F1-score
- Accuracy

| Modèle | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| LayoutLMv3 | 0.5952 | 0.7653 | **0.6696** | 0.9852 |
| LiLT | À compléter | À compléter | À compléter | À compléter |

**LayoutLMv3** est retenu comme modèle de référence.

---

## Inférence

Le modèle LayoutLMv3 fine-tuné est publié sur Hugging Face et chargé par l'application :

`djamali/layoutlmv3-photovoltaic`

Le pipeline d'inférence est :

```text
PDF
 ↓
Tesseract
 ↓
Tokens + Bounding Boxes
 ↓
LayoutLMv3
 ↓
Token Classification
 ↓
Agrégation des entités
 ↓
JSON
```

Exemple de réponse :

```json
{
  "client": "...",
  "fournisseur": "...",
  "numero_devis": "...",
  "date_devis": "...",
  "montant_total": "..."
}
```

---

## Architecture

```text
document-intelligence-engine/
├── app/
│   ├── main.py                 # API FastAPI
│   ├── model.py                # Chargement du modèle et inférence
│   ├── preprocess.py           # Prétraitement des PDF
│   └── streamlit_app.py        # Interface de démonstration
│
├── scripts/
│   ├── ocr_benchmark/          # Benchmark OCR
│   ├── data_preparation/       # Images, Label Studio et dataset
│   ├── training/               # Fine-tuning LayoutLMv3 et LiLT
│   ├── evaluation/             # Évaluation finale
│   ├── utils/                  # Fonctions partagées
│   └── archive/                # Code historique
│
├── notebooks/                  # Exploration et expérimentations
├── tests/                      # Tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-runtime.txt
```

Les documents sources, datasets générés et modèles entraînés ne sont pas versionnés dans Git.

---

## Installation

Créer et activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Lancer l'application en local

L'application peut être testée localement sans Docker. Elle comporte deux processus à lancer dans deux terminaux distincts.

Dans un premier terminal, activer l'environnement virtuel, définir le token Hugging Face puis démarrer l'API FastAPI :

```bash
source .venv/bin/activate
export HF_TOKEN=<token>
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API est alors accessible à l'adresse :

```text
http://localhost:8000
```

La documentation interactive de l'API est disponible à l'adresse :

```text
http://localhost:8000/docs
```

Dans un second terminal, activer le même environnement puis démarrer l'interface Streamlit :

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8501
```

Ouvrir ensuite l'application de démonstration dans le navigateur :

```text
http://localhost:8501
```

Il est alors possible de charger un devis PDF afin de tester son traitement par Tesseract et LayoutLMv3, puis de visualiser les informations extraites.

---

## Exécution avec Docker

Le projet expose deux services :

- **FastAPI** sur le port `8000`
- **Streamlit** sur le port `8501`

Le modèle est récupéré depuis Hugging Face au démarrage de l'API.

Définir le token Hugging Face :

```bash
export HF_TOKEN=<token>
```

Puis lancer :

```bash
docker compose up
```

API :

```text
http://localhost:8000
```

Interface Streamlit :

```text
http://localhost:8501
```

Vérification de l'API :

```text
GET /health
```

Prédiction :

```text
POST /predict
Content-Type: application/pdf
```

---

## Reproduire le pipeline

Les principales étapes sont organisées dans `scripts/` :

```text
Benchmark OCR
      ↓
Génération des images
      ↓
Préparation des tâches Label Studio
      ↓
Annotation
      ↓
Construction du dataset
      ↓
Fine-tuning LayoutLMv3
      ↓
Fine-tuning LiLT
      ↓
Évaluation finale
```

Scripts principaux :

```text
scripts/ocr_benchmark/step_1_benchmark_tesseract_ocr.py
scripts/data_preparation/step_2_generate_images.py
scripts/data_preparation/step_3_prepare_labelstudio_tasks.py
scripts/data_preparation/step_4_build_layoutlm_dataset.py
scripts/training/step_5_train_layoutlmv3.py
scripts/training/step_6_train_lilt.py
scripts/evaluation/step_7_evaluate_layoutlmv3_test.py
scripts/evaluation/step_7_evaluate_lilt_test.py
```

---

## Démonstration

L'application de démonstration permet de charger un devis PDF depuis Streamlit.

```text
Utilisateur
    ↓
Streamlit
    ↓
FastAPI
    ↓
Tesseract
    ↓
LayoutLMv3
    ↓
JSON métier
```

Le projet peut également être exécuté dans GitHub Codespaces avec Docker Compose pour disposer d'une démonstration accessible à distance.

---

## Limites et perspectives

- dataset encore limité à 150 documents ;
- dépendance à la qualité de l'OCR ;
- variabilité importante des documents entre fournisseurs ;
- généralisation à confirmer sur davantage de fournisseurs ;
- performances à améliorer avant une utilisation en production ;
- enrichissement futur du dataset et optimisation du modèle.
