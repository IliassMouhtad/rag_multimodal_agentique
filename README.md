# 🤖 RAG Multimodal & Agentique (Textes & Images)

Ce projet implémente un système complet de **RAG (Retrieval-Augmented Generation) Multimodal et Agentique** capable d'extraire, indexer, retrouver et raisonner de manière autonome sur du texte et des images issus du papier de recherche séminal :
> **"Attention Is All You Need"** (*Vaswani et al., 2017*).

---

## 🏛️ Architecture Globale (Pattern ReAct & LangGraph)

```
   ┌────────────────────────────────────────────────────────┐
   │             Utilisateur / Requête Multimodale          │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             MIND (Cerveau - Raisonnement LLM)           │
   │      - Analyse la question                             │
   │      - Formule une Thought (Pensée)                    │
   │      - Décide d'une Action (tool call: retriever_tool) │
   └───────────┬────────────────────────────────▲───────────┘
               │                                │
     (Action : appel d'outil)            (Observation)
               │                                │
               ▼                                │
   ┌────────────────────────────────────────────┴───────────┐
   │             BODY (Corps - ToolNode)                    │
   │  Exécute retriever_tool sur ChromaDB (CLIP ViT-B-32)    │
   └────────────────────────────────────────────────────────┘
               │ (Synthèse finale prête)
               ▼
   ┌────────────────────────────────────────────────────────┐
   │             END (Réponse Finale Documentée)            │
   │  Citations précises [Attention is all you need, p. X]  │
   │  et rendu visuel des diagrammes associés               │
   └────────────────────────────────────────────────────────┘
```

---

## 📋 Les 7 Étapes du Projet

1. **Étape 1 : Extraction du contenu multimodal (Textes & Images)**
   - Extraction ordonnée du texte page par page avec `pymupdf` (`fitz`).
   - Extraction des figures intégrées par identifiants XREF (Figure 1 de l'architecture Transformer et Figures 2 d'attention).
2. **Étape 2 : Vectorisation dans un espace latent commun (CLIP)**
   - Modèle vision-langage `ViT-B-32` (`open_clip`, poids `laion2b_s34b_b79k`).
   - Découpage en fragments de 60 tokens BPE (chevauchement 10).
   - Normalisation $\ell_2$ stricte : $\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$.
3. **Étape 3 : Indexation et stockage dans ChromaDB**
   - Base vectorielle persistante avec métrique cosinus (`{"hnsw:space": "cosine"}`).
   - Gestion de l'idempotence via hachage SHA-256 du document.
   - Métadonnées complètes (`type`, `pdf`, `page`, `fragment`, `source`).
4. **Étape 4 : Recherche multimodale par similarité cosinus**
   - 4 modes : Texte $\to$ Texte, Texte $\to$ Image, Image $\to$ Image, Image $\to$ Texte.
   - Similarité : $S = 1 - \text{distance}$.
5. **Étape 5 : Outil Agentique (`retriever_tool`)**
   - Décorateur `@tool` avec annotations strictes `Literal["tous", "texte", "image"]` et schéma de description JSON pour le LLM.
6. **Étape 6 : Construction de l'Agent IA (ReAct & LangGraph)**
   - Articulation formelle MIND (Raisonnement) et BODY (ToolNode).
   - Compilation du graphe d'états `MessagesState` via `langgraph.graph.StateGraph` et `tools_condition`.
7. **Étape 7 : Tests et validation empirique**
   - Validation sur requêtes hybrides et requêtes basées sur une image en entrée.
   - Analyse dynamique des tool calls et de la fidélité documentaire.

---

## 📁 Structure des Fichiers

| Fichier / Dossier | Description |
| :--- | :--- |
| `TP_RAG_Multimodal_Agentique.ipynb` | **Nouveau notebook complet** détaillant pas à pas les 7 étapes avec code et explications. |
| `rag_multimodal_agentique.ipynb` | Notebook de cours initial (conservé comme source d'inspiration). |
| `rag_backend.py` | Module backend réutilisable (extraction, CLIP, ChromaDB, tool, agent LangGraph). |
| `app_streamlit.py` | Interface web interactive pour interroger l'agent et tester la recherche vectorielle. |
| `attention is all you need.pdf` | Document PDF source analysé. |
| `architecture_transformer.png` | Schéma extrait de l'architecture Transformer (Figure 1). |
| `resultats_pdf/` | Dossier contenant les textes extraits par page et les figures PNG. |
| `chroma_pdf/` | Base vectorielle persistante ChromaDB avec index HNSW. |

---

## 🚀 Installation & Exécution

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
# ou individuellement :
pip install chromadb open-clip-torch pymupdf torch torchvision langchain langgraph langchain-openai streamlit
```

### 2. Exécution du Notebook

Ouvrez et exécutez le notebook dans Jupyter :
```bash
jupyter notebook TP_RAG_Multimodal_Agentique.ipynb
```

### 3. Lancement de l'Application Streamlit

Pour lancer l'interface interactive :
```bash
streamlit run app_streamlit.py
```
