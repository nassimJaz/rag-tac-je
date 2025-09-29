# Projet RAG (Retrieval-Augmented Generation)

Ce projet est un chatbot conçu spécifiquement pour les Junior-Entreprises. Sa mission est de répondre à leurs questions complexes, qu'elles portent sur le cadre légal, les processus internes, la trésorerie ou les normes qualité. L'application fonctionne sur le principe du RAG (Retrieval-Augmented Generation) : elle ingère un ensemble de documents de référence (normes, guides, documents légaux), les analyse et les stocke. Lorsqu'une question est posée, le chatbot recherche les informations les plus pertinentes dans sa base de connaissances et s'appuie sur un modèle de langage pour formuler une réponse précise et contextualisée.

## Questions pertinantes

Voici une liste de question pour tester le RAG rapidement :

- Qu'est ce qu'un JEH ?

- Est ce qu'un RM est obligatoire ?

- Décris moi l'architecture d'un Junior Entreprise.

- Parle moi du mouvement des Juniors Entreprises.

- Est ce qu'il est possible de faire des sous traitances au sein d'un Junior Entreprise ?

- Donne moi les documents obligatoires d'un étude.

## Utilisation et installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <url-du-depot>
    cd <nom-du-depot>
    ```

2.  **Lancer le service par docker:**
    ```bash
    docker compose up -d
    ```
3. **Streamlit :**
    Aller sur le port `http://localhost:8501`
    Lors de la première utilisation lancer l'ingestion et échauffer le modèle d'embedding avec une question.
    L'installation de l'embedding sur le service peut prendre jusqu'à 5min.

## Clé Portkey :

Voici un guide simple pour avoir une clé Portkey. Portkey est un proxy-multi LLM.
Il est donc possible de choisir tout type de LLM disponible sur Portkey tel que OpenAI, Claude Code, Open Router, Groq etc ...

Il est important de choisir le modèle en fonction du fournisseur dans le .env

1. Se rendre sur le site de [Portkey](https://portkey.ai/)
2. Se connecter/inscrire
3. Ajouter un fournisseur et le choisir : [](https://app.portkey.ai/model-catalog/providers)
4. Cliquer sur : +add new integration
5. Donner le nom voulu
6. Mettre pour le slug : **rag_llm** ou bien en mettre un personnalisé à changer dans le .env :
```bash
SLUG_PORTKEY=
```
7. Mettre la clé API du fournisseur
8. Accéder aux modèles disponibles sur Portkey: [Models](https://app.portkey.ai/model-catalog)
9. Voir "models" et le copier dans le .env :
```bash
GENERATION_MODEL=
```

9. Générer sa clé API Portkey. Se rendre sur : [API Keys](https://app.portkey.ai/api-keys)
10. Cliquer sur : Create (et laisser sur service pour avoir tout les droits)
11. Cliquer sur create en bas à droite
12. Copier et coller la clé générée dans le .env : 
```bash
PORTKEY_KEY=
```

## Configuration

Créez un fichier `.env` à la racine du projet pour configurer l'application à l'aide du `.env.example`
```bash
touch .env
```

Voici la liste des variables d'environnement que vous pouvez définir :

| Variable | Description | Défaut |
| --- | --- | --- |
| `EMBEDDING_MODEL` | Modèle d'embedding à utiliser. | `sentence-transformers/distiluse-base-multilingual-cased-v2` |
| `EMBED_DIM` | Dimension des embeddings. | `512` |
| `CHUNK_SIZE` | Taille des chunks pour le découpage des documents. | `1024` |
| `CHUNK_OVERLAP` | Chevauchement entre les chunks. | `300` |
| `USE_GPU` | Mettre à `true` pour utiliser le GPU. | `false` |
| `VECTOR_STORE_DIR`| Dossier pour la base de données vectorielle. | `./qdrant` |
| `RAG_METHOD` | Méthode de RAG (`similarity` ou `hyde`). | `similarity` |
| `FILE_DIR` | Dossier contenant les fichiers à ingérer (PDF, CSV, JSON). | `./docs` |
| `PROMPTS_DIR` | Dossier contenant les prompts du RAG en format `.j2` | `./app/prompts` |
| `CSV_DELIMITER` | Délimiteur pour les fichiers CSV. | `,` |
| `PROVIDER` | Fournisseur de LLM (`ollama` ou `portkey`). | `ollama` |
| `GENERATION_MODEL`| Modèle de génération à utiliser. | `meta-llama/llama-3.3-70b-instruct:free` |
| `HYDE_MODEL` | Modèle pour la génération de document hypothétique (HyDE). | `mistralai/devstral-small-2505:free` |
| `PORTKEY_KEY` | Clé API pour Portkey. | |
| `SLUG_PORTKEY` | Slug pour Portkey. | `rag_llm` |
| `OLLAMA_HOST` | Hôte pour Ollama. | `http://localhost:11434` |
| `TEMPERATURE` | Température pour la génération de la réponse. | `0` |
| `TOP_K` | Nombre de documents à récupérer parmi les k plus probables. | `10` |
| `RERANKER_ENABLE` | Activer le reranker. | `true` |
| `RERANKER_TOP_K` | Nombre de documents à garder après le reranking. | `3` |
| `CROSS_ENCODER` | Modèle de cross-encoder à utiliser pour le reranking. | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

## Stack Technique

Ce projet utilise les technologies suivantes :

-   **Framework RAG**: [Haystack](https://haystack.deepset.ai/) pour l'orchestration de la pipeline de Retrieval-Augmented Generation.
-   **Base de Données Vectorielle**: [Qdrant](https://qdrant.tech/) pour le stockage et la recherche des embeddings de documents.
-   **Modèles d'Embedding**: [SentenceTransformers](https://www.sbert.net/) pour la création des représentations vectorielles des textes.
-   **Fournisseurs de LLM**:
    -   [Ollama](https://ollama.ai/) pour l'exécution de modèles de langage en local.
    -   [Portkey](https://portkey.ai/) comme passerelle pour accéder à divers fournisseurs de LLM (OpenAI, Groq, etc.).
-   **Traitement de Documents**:
    -   `pypdf` pour l'extraction de texte à partir de fichiers PDF.
    -   `langdetect` pour la détection de la langue des documents.
-   **Templating de Prompt**: `Jinja2` pour la construction dynamique des prompts.
-   **Configuration**: `python-dotenv` pour la gestion des variables d'environnement.
