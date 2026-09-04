
# Document Intelligence Engine

Extraction structurée d'informations à partir de devis photovoltaïques avec OCR et modèles Document AI.

Projet réalisé dans le cadre de la formation **Développeur IA d'Alyra**, à partir d'un cas d'usage fourni par **Beluo**.

Le système transforme un devis PDF en données structurées en combinant :

- OCR avec Tesseract ;
- compréhension du document avec LayoutLMv3 ou LiLT ;
- classification de tokens avec annotations BIO ;
- API FastAPI ;
- interface de démonstration Streamlit ;
- conteneurisation Docker.

---

## Objectif

Les devis photovoltaïques sont des documents semi-structurés dont la présentation varie selon les fournisseurs.

L'objectif du projet est d'extraire automatiquement cinq informations :

- émetteur du devis ;
- client ;
- numéro du devis ;
- date d'émission ;
- montant total.

La sortie du système est une structure JSON exploitable par une application métier.

Exemple :

```json
{
  "emetteur_devis": "...",
  "client": "...",
  "numero_devis": "...",
  "date_devis": "...",
  "montant_total": "..."
}
````

---

## Architecture

Le pipeline final est le suivant :

```text
             PDF
              |
              v
      Conversion en images
              |
              v
       OCR avec Tesseract
              |
              v
   Tokens + bounding boxes
              |
              v
       Modèle Document AI
   LayoutLMv3 / LiLT
              |
              v
 Classification de tokens BIO
              |
              v
    Agrégation des entités
              |
              v
             JSON
```

Pour la démonstration :

```text
Utilisateur
    |
    v
Streamlit
    |
    v
FastAPI
    |
    v
Prétraitement / OCR
    |
    v
LayoutLMv3
    |
    v
Post-traitement
    |
    v
JSON structuré
```

---

## Dataset

Le dataset est constitué de **150 devis photovoltaïques réels fournis par Beluo**.

Les documents proviennent de plusieurs formats de devis et présentent des variations de mise en page représentatives du cas d'usage étudié.

### Entités annotées

Cinq types d'entités sont recherchés :

| Entité           | Description                   |
| ---------------- | ----------------------------- |
| `CLIENT`         | Nom ou identité du client     |
| `DATE_DEVIS`     | Date d'émission du devis      |
| `EMETTEUR_DEVIS` | Société ou organisme émetteur |
| `MONTANT_TOTAL`  | Montant total du devis        |
| `NUMERO_DEVIS`   | Numéro ou référence du devis  |

Les annotations utilisent le format **BIO** :

* `B-*` : début d'une entité ;
* `I-*` : continuation d'une entité ;
* `O` : token ne faisant partie d'aucune entité.

Avec cinq entités, le problème comporte **11 classes** :

```text
B-CLIENT
I-CLIENT
B-DATE_DEVIS
I-DATE_DEVIS
B-EMETTEUR_DEVIS
I-EMETTEUR_DEVIS
B-MONTANT_TOTAL
I-MONTANT_TOTAL
B-NUMERO_DEVIS
I-NUMERO_DEVIS
O
```

---

## Annotation

Les documents ont été annotés avec **Label Studio**.

Pour chaque document, les annotations permettent d'associer :

* le texte ;
* les tokens ;
* leur position dans le document ;
* leur label BIO.

Ces informations sont ensuite utilisées pour préparer les données d'entrée des modèles Document AI.

---

## Séparation train / validation / test

Le dataset a été séparé **au niveau du document**, et non au niveau de la page.

Cette séparation évite qu'une page provenant d'un même devis se retrouve, par exemple, dans le jeu d'entraînement et dans le jeu de test.

| Split      | Documents |   Pages |
| ---------- | --------: | ------: |
| Train      |       105 |     148 |
| Validation |        22 |      31 |
| Test       |        23 |      35 |
| **Total**  |   **150** | **214** |

Répartition :

* Train : 70 %
* Validation : 15 %
* Test : 15 %

Le jeu de test reste indépendant pendant l'entraînement et est utilisé pour l'évaluation finale.

---

## OCR

Plusieurs moteurs OCR ont été étudiés :

* Tesseract ;
* EasyOCR ;
* PaddleOCR.

**Tesseract** a été retenu pour le pipeline final.

L'OCR fournit notamment :

* le texte reconnu ;
* les tokens ;
* les coordonnées spatiales des tokens.

Les coordonnées sont ensuite normalisées afin d'être utilisées par les modèles Document AI.

---

## Modèles évalués

Deux architectures pré-entraînées ont été fine-tunées pour la classification de tokens.

### LayoutLMv3

Modèle de base :

```text
microsoft/layoutlmv3-base
```

LayoutLMv3 exploite conjointement les informations :

* textuelles ;
* spatiales ;
* visuelles.

### LiLT

Modèle utilisé :

```text
SCUT-DLVCLab/lilt-roberta-en-base
```

LiLT exploite notamment le texte et la structure spatiale du document.

L'objectif de l'expérimentation était de comparer les deux approches sur le même problème d'extraction.

---

## Fine-tuning

Le problème est traité comme une tâche de **token classification**.

Le pipeline d'entraînement comprend notamment :

1. chargement du dataset ;
2. encodage des tokens et bounding boxes ;
3. mapping des labels BIO ;
4. séparation train / validation / test par document ;
5. fine-tuning ;
6. sélection du meilleur checkpoint sur le jeu de validation ;
7. évaluation finale sur le jeu de test indépendant.

La métrique principale utilisée pour comparer les modèles est le **F1-score**.

---

## Résultats

Les résultats ci-dessous correspondent à l'évaluation finale sur le **jeu de test indépendant** de 23 documents / 35 pages.

| Modèle         |   Precision |      Recall |    F1-score |    Accuracy |
| -------------- | ----------: | ----------: | ----------: | ----------: |
| **LayoutLMv3** | **58.16 %** | **83.67 %** | **68.62 %** | **98.35 %** |
| LiLT           |     52.52 % |     74.49 % |     61.60 % |     98.27 % |

### Modèle retenu

**LayoutLMv3** est retenu comme modèle de référence.

Son F1-score sur le jeu de test atteint :

```text
68.62 %
```

contre :

```text
61.60 %
```

pour LiLT.

L'écart est donc de **7.02 points de F1** en faveur de LayoutLMv3.

LayoutLMv3 obtient également un rappel de **83.67 %**, contre **74.49 %** pour LiLT.

### Interprétation de l'accuracy

L'accuracy dépasse 98 % pour les deux modèles, mais cette métrique doit être interprétée avec prudence.

La majorité des tokens d'un document appartient à la classe `O` (hors entité). Une accuracy élevée ne signifie donc pas nécessairement que toutes les entités métier sont correctement extraites.

Pour cette raison, le **F1-score constitue la métrique principale de comparaison**.

---

## Résultat détaillé LayoutLMv3

Évaluation finale :

```text
test_loss      : 0.1026
test_precision : 0.5816
test_recall    : 0.8367
test_f1        : 0.6862
test_accuracy  : 0.9835
```

Ces métriques correspondent au modèle sélectionné après entraînement et évalué sur le jeu de test indépendant.

---

## Modèle publié

Le modèle LayoutLMv3 fine-tuné est disponible sur Hugging Face :

```text
djamali/layoutlmv3-photovoltaic
```

Il correspond au modèle spécialisé dans l'extraction des entités des devis photovoltaïques étudiés dans ce projet.

---

## API FastAPI

Le modèle est intégré dans une API **FastAPI**.

Principaux endpoints :

```text
GET /health
POST /predict
```

### `/health`

Permet de vérifier que le service est disponible.

### `/predict`

Reçoit un document et exécute le pipeline :

```text
document
   ↓
OCR
   ↓
prétraitement
   ↓
LayoutLMv3
   ↓
classification
   ↓
agrégation des entités
   ↓
JSON
```

---

## Interface Streamlit

Une interface **Streamlit** permet de tester le pipeline sans appeler directement l'API.

Elle permet :

1. de sélectionner un devis ;
2. de lancer l'analyse ;
3. d'envoyer le document à FastAPI ;
4. d'exécuter l'OCR et le modèle ;
5. d'afficher les informations extraites.

Cette interface a été utilisée pour la démonstration finale du projet.

---

## Structure du projet

```text
document-intelligence-engine/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── preprocess.py
│
├── benchmarks/
│   └── ocr/
│
├── dataset/
│
├── label_studio_data/
│
├── notebooks/
│
├── scripts/
│
├── tests/
│
├── DEMO.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-runtime.txt
└── README.md
```

---

## Installation

Cloner le repository :

```bash
git clone https://github.com/djamaliAliBahaBakar/document-intelligence-engine.git
cd document-intelligence-engine
```

Créer un environnement Python et installer les dépendances :

```bash
pip install -r requirements.txt
```

Tesseract doit également être installé sur le système.

---

## Lancer l'API

```bash
uvicorn app.main:app --reload
```

L'API est alors disponible sur :

```text
http://localhost:8000
```

Documentation OpenAPI :

```text
http://localhost:8000/docs
```

---

## Lancer Streamlit

Dans un second terminal :

```bash
streamlit run app/streamlit_app.py
```

L'interface est alors accessible sur :

```text
http://localhost:8501
```

> Le chemin exact du fichier Streamlit dépend de la structure présente dans le repository. Vérifier le nom du fichier avant d'utiliser cette commande.

---

## Docker

Le projet contient :

* un `Dockerfile` ;
* un fichier `docker-compose.yml`.

Le pipeline peut être lancé avec Docker Compose :

```bash
docker compose up --build
```

Cela permet d'exécuter l'environnement de démonstration de manière reproductible.

---

## Tests

Le repository contient des tests automatisés dans :

```text
tests/
```

Ils permettent notamment de vérifier les composants du pipeline et l'intégration de l'application.

Exécution :

```bash
pytest
```

Le nombre exact de tests réussis n'est volontairement pas indiqué ici afin d'éviter de publier une valeur qui n'a pas été vérifiée sur la version finale du repository.

---

## Limites

Ce projet constitue un prototype fonctionnel de Document AI et non un système prêt pour une utilisation industrielle.

Les principales limites identifiées sont :

### Taille du dataset

Le dataset contient 150 documents.

Cette taille permet de construire et d'évaluer le pipeline, mais reste limitée pour garantir une généralisation à l'ensemble des formats de devis existants.

### Variabilité documentaire

Les fournisseurs utilisent des structures, vocabulaires et mises en page différents.

Les performances peuvent donc varier sur des formats très différents de ceux présents dans le dataset.

### Dépendance à l'OCR

Une erreur de reconnaissance produite par l'OCR peut se propager jusqu'au modèle de classification.

### Précision du modèle

LayoutLMv3 obtient un rappel élevé de 83.67 %, mais une précision de 58.16 %.

Le système retrouve donc une part importante des entités attendues, mais produit encore des faux positifs.

### Généralisation

Les performances publiées correspondent au jeu de test constitué à partir du corpus Beluo.

Une validation sur de nouveaux fournisseurs et de nouveaux documents serait nécessaire avant toute utilisation en production.

---

## Améliorations possibles

Plusieurs pistes pourraient être étudiées :

* enrichissement du dataset ;
* ajout de documents provenant de nouveaux fournisseurs ;
* amélioration du post-traitement des entités ;
* analyse des erreurs par type de champ ;
* optimisation des hyperparamètres ;
* comparaison avec des modèles Vision-Language récents ;
* extraction de structures plus complexes, notamment les tableaux.

Ces éléments constituent des pistes d'évolution et ne sont pas présentés comme des fonctionnalités déjà implémentées.

---

## Ce que démontre ce projet

Ce projet couvre un cycle Document AI complet :

```text
Données réelles
    ↓
Annotation
    ↓
OCR
    ↓
Préparation du dataset
    ↓
Fine-tuning
    ↓
Évaluation
    ↓
Comparaison de modèles
    ↓
Sélection du modèle
    ↓
API
    ↓
Interface de démonstration
    ↓
Conteneurisation
```

Il met notamment en œuvre :

* préparation et annotation de données documentaires ;
* classification de tokens avec BIO ;
* OCR ;
* Transformers ;
* LayoutLMv3 ;
* LiLT ;
* fine-tuning ;
* évaluation sur un jeu de test indépendant ;
* FastAPI ;
* Streamlit ;
* Docker.

---

## Contexte

Projet réalisé dans le cadre de la formation **Développeur IA — Alyra**.

Le cas d'usage et les **150 devis photovoltaïques réels** utilisés pour constituer le dataset ont été fournis par **Beluo**.

L'objectif était de mettre en œuvre un pipeline complet de Deep Learning appliqué à la compréhension de documents, depuis les données et leur annotation jusqu'à l'exposition du modèle dans une application fonctionnelle.

---

## Auteur

**Djamali Ali Baha Bakar**

Senior Software Engineer / Tech Lead — Applied AI & AI-enabled applications

