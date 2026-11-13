import streamlit as st
import os
import json
from pathlib import Path
from PIL import Image

from rag_backend import (
    DEFAULT_PDF,
    extraire_contenu_pdf,
    MultimodalVectorStore,
    creer_agent_langgraph,
    interroger_rag_agentique,
    retriever_tool
)

st.set_page_config(
    page_title="RAG Multimodal Agentique - Attention Is All You Need",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Multimodal & Agentique (Textes & Images)")
st.caption("Conçu sur l'article *'Attention Is All You Need'* (Vaswani et al.) avec CLIP (ViT-B-32), ChromaDB et LangGraph (ReAct).")

# Initialisation du Store en cache
@st.cache_resource
def get_store():
    return MultimodalVectorStore()

store = get_store()

# Sidebar : Configuration & Gestion du Document
with st.sidebar:
    st.header("📄 Source Documentaire")
    st.info(f"Document : **{DEFAULT_PDF.name}**")
    
    st.markdown("---")
    st.subheader("🔑 Modèle LLM de l'Agent")
    api_key_input = st.text_input("OpenAI API Key (optionnel)", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    
    st.caption("Si aucune clé n'est renseignée, un moteur ReAct local autonome assure la boucle de raisonnement.")

    st.markdown("---")
    st.subheader("⚙️ Maintenance Indexation")
    if st.button("🔄 Réindexer le PDF complet"):
        with st.spinner("Extraction et vectorisation en cours..."):
            extraire_contenu_pdf(DEFAULT_PDF)
            nb_txt, nb_img = store.indexer_pdf(DEFAULT_PDF)
            st.success(f"Indexation réussie : {nb_txt} fragments de texte, {nb_img} images.")

    count = store.collection.count()
    st.metric("Total éléments indexés (ChromaDB)", count)

# Onglets principaux
tab_agent, tab_retrieval, tab_architecture = st.tabs([
    "🧠 Agent Multimodal (ReAct & LangGraph)",
    "🔍 Recherche Vectorielle Directe",
    "📐 Architecture & Détails Techniques"
])

# ==========================================
# ONGLET 1 : AGENT MULTIMODAL (ReAct)
# ==========================================
with tab_agent:
    st.markdown("### 💬 Poser une question à l'Agent ReAct")
    st.write(
        "L'agent analyse votre question, décide d'appeler l'outil `retriever_tool` avec les paramètres adéquats, "
        "observe les fragments ou diagrammes retrouvés, puis synthétise une réponse documentée."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        question_agent = st.text_area(
            "Votre question :",
            value="Explique le fonctionnement de l'attention multi-têtes et retrouve le diagramme de la figure 2.",
            height=100
        )
    with col2:
        img_file = st.file_uploader("Image de requête (optionnel) :", type=["png", "jpg", "jpeg"])
        img_temp_path = None
        if img_file is not None:
            img = Image.open(img_file)
            st.image(img, caption="Image fournie", use_container_width=True)
            img_temp_path = Path("temp_query_image.png")
            img.save(img_temp_path)

    if st.button("🚀 Interroger l'Agent", type="primary"):
        with st.spinner("Cycle ReAct (MIND ⇄ BODY) en cours..."):
            agent = creer_agent_langgraph()
            resultat = interroger_rag_agentique(
                agent,
                question=question_agent,
                image_requete=img_temp_path,
                verbose=False
            )

            st.markdown("#### 📝 Réponse Finale de l'Agent")
            st.markdown(resultat["reponse"])

            if resultat["images"]:
                st.markdown("#### 🖼️ Figures Retrouvées")
                cols = st.columns(len(resultat["images"]))
                for idx, im in enumerate(resultat["images"]):
                    with cols[idx]:
                        st.image(str(im["path"]), caption=f"Page {im['page']} (Sim: {im['similarite']})", use_container_width=True)

            with st.expander("🔍 Trace d'exécution ReAct (Raisonnement & Outils)"):
                st.write("**Appels d'outils effectués par l'agent :**")
                for appel in resultat["appels_outils"]:
                    st.json(appel)
                st.write("**Historique complet des messages (MessagesState) :**")
                for msg in resultat["execution"]["messages"]:
                    st.text(f"[{type(msg).__name__}]: {msg.content}")

# ==========================================
# ONGLET 2 : RECHERCHE VECTORIELLE DIRECTE
# ==========================================
with tab_retrieval:
    st.markdown("### 🔍 Test Direct du Retriever Multimodal (Étape 4)")
    st.write("Interrogez directement l'espace latent CLIP sans passer par l'agent.")

    mode_requete = st.radio("Type de requête :", ["Texte", "Image"], horizontal=True)
    c_type = st.selectbox("Filtrer les résultats :", ["tous", "texte", "image"])
    k_val = st.slider("Nombre de résultats (k) :", min_value=1, max_value=10, value=4)

    filtre_type = None if c_type == "tous" else c_type

    if mode_requete == "Texte":
        texte_req = st.text_input("Requête textuelle :", value="Scaled Dot-Product Attention equation and diagram")
        if st.button("Rechercher (Texte)"):
            with st.spinner("Recherche cosinus..."):
                res = store.rechercher(texte=texte_req, type_resultat=filtre_type, k=k_val)
                st.markdown("#### Résultats :")
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                    sim = round(1.0 - dist, 3)
                    with st.container():
                        st.markdown(f"**[{meta['type'].upper()}] - Page {meta['page']} (Similarité : {sim})**")
                        if meta["type"] == "texte":
                            st.write(doc)
                        else:
                            st.image(meta["source"], width=450)
                        st.caption(f"Source : {meta['source']}")
                        st.markdown("---")
    else:
        file_req = st.file_uploader("Charger une image pour la recherche :", type=["png", "jpg", "jpeg"], key="direct_img")
        if file_req is not None:
            st.image(file_req, width=300, caption="Image requête")
            if st.button("Rechercher par Image"):
                temp_p = Path("temp_direct_query.png")
                with open(temp_p, "wb") as f:
                    f.write(file_req.getbuffer())
                with st.spinner("Calcul embedding CLIP image et recherche..."):
                    res = store.rechercher(image=temp_p, type_resultat=filtre_type, k=k_val)
                    st.markdown("#### Résultats :")
                    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                        sim = round(1.0 - dist, 3)
                        st.markdown(f"**[{meta['type'].upper()}] - Page {meta['page']} (Similarité : {sim})**")
                        if meta["type"] == "texte":
                            st.write(doc)
                        else:
                            st.image(meta["source"], width=450)
                        st.caption(f"Source : {meta['source']}")
                        st.markdown("---")

# ==========================================
# ONGLET 3 : ARCHITECTURE TECHNIQUE
# ==========================================
with tab_architecture:
    st.markdown("""
    ### 📐 Architecture du Système RAG Multimodal & Agentique
    
    1. **Extraction Multimodale (Étape 1)** :
       - Découpage par page et parsing PDF avec `PyMuPDF` (`fitz`).
       - Extraction automatique des images intégrées (XREF) et sauvegarde avec nommage indexé (`page_XXX_image_YYY.png`).
    
    2. **Espace Latent Partagé CLIP (Étape 2)** :
       - Modèle : `ViT-B-32` préentraîné sur `laion2b_s34b_b79k` (`open_clip`).
       - Les textes (découpés en fenêtres adaptées de 60 tokens BPE) et les images sont projetés dans le même espace vectoriel à 512 dimensions.
       - Normalisation L2 stricte : $\\mathbf{v}_{norm} = \\frac{\\mathbf{v}}{\\|\\mathbf{v}\\|_2}$.
    
    3. **Indexation ChromaDB (Étape 3)** :
       - Collection persistante avec espace métrique cosinus (`hnsw:space: cosine`).
       - Gestion du cycle de vie documentaire via hachage SHA-256 du PDF pour réindexation idempotent.
    
    4. **Recherche Multimodale (Étape 4)** :
       - Requête texte $\\to$ textes et/ou images.
       - Requête image $\\to$ textes et/ou images.
       - Similarité cosinus : $S = 1 - \\text{distance}$.
    
    5. **Outil Agentique `retriever_tool` (Étape 5)** :
       - Encapsulation standard LangChain `@tool` avec typage `Literal` et schéma JSON de description pour guider le LLM.
    
    6. **Agent ReAct & LangGraph (Étape 6 & 7)** :
       - **MIND (Cerveau)** : Noeud LLM de raisonnement et planification.
       - **BODY (Corps)** : Noeud `ToolNode` exécutant les actions dans la base vectorielle.
       - **Cycle ReAct** : *Thought $\\to$ Action $\\to$ Observation $\\to$ Final Synthesis*.
       - Graphe d'état `MessagesState` compilé via `langgraph.graph.StateGraph`.
    """)

