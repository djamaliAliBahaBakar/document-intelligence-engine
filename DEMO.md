Oui. Avec ce qu’on vient de mettre en place, la procédure sera assez courte.

### Procédure

1. Va sur [GitHub Codespaces](https://github.com/codespaces?utm_source=chatgpt.com).

2. Retrouve le Codespace de `document-intelligence-engine` et clique dessus pour le démarrer. **Ne crée pas un nouveau Codespace** à chaque démonstration.

3. Une fois VS Code ouvert, vérifie éventuellement que le secret Hugging Face est disponible :

```bash
echo ${HF_TOKEN:+HF_TOKEN présent}
```

4. Lance l'application :

```bash
docker compose up
```

Comme l'image Docker a déjà été construite dans ce Codespace, tu ne devrais normalement **pas avoir besoin de `--build`** tant que tu n'as pas modifié le code ou les dépendances.

5. Attends dans les logs :

```text
Chargement du modèle depuis : djamali/layoutlmv3-photovoltaic
Modèle chargé.
```

et le démarrage de Streamlit.

6. Dans l'onglet **PORTS**, repère :

```text
8501
```

Passe sa visibilité en **Public** si elle est revenue en Private.

7. Ouvre l'URL `8501` et fais toi-même un test avec un PDF.

8. Si tout fonctionne, copie **cette URL Streamlit** et transmets-la à ton chef de projet/jury.

La personne pourra alors faire :

```text
URL publique
    ↓
Streamlit
    ↓
Upload PDF
    ↓
FastAPI
    ↓
Tesseract
    ↓
LayoutLMv3
    ↓
JSON
```

Elle n'a besoin ni de Docker, ni de Python, ni de GitHub, ni de Hugging Face.

### Quand tu as terminé la démonstration

Tu peux arrêter :

```bash
docker compose down
```

puis arrêter le Codespace depuis GitHub pour ne pas consommer inutilement ton quota Codespaces.

### Si tu as modifié le projet entre-temps

Il faudra d'abord récupérer la dernière version :

```bash
git pull
```

et, si les modifications concernent le Dockerfile, les dépendances ou l'application :

```bash
docker compose up --build
```

Sinon :

```bash
docker compose up
```



