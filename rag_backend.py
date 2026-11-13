import os
import re
import json
import base64
import hashlib
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple, Literal

import torch
import open_clip
from open_clip.tokenizer import SimpleTokenizer
import chromadb
import pymupdf
from PIL import Image

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- Configuration globale ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CHROMA_DIR = Path("chroma_pdf")
DEFAULT_PDF = Path("attention is all you need.pdf")
if not DEFAULT_PDF.is_file() and Path("pdfs/attention is all you need.pdf").is_file():
    DEFAULT_PDF = Path("pdfs/attention is all you need.pdf")

OUTPUT_DIR = Path("resultats_pdf")
TEXT_DIR = OUTPUT_DIR / "textes"
IMAGE_DIR = OUTPUT_DIR / "images"

# --- Étape 1 : Extraction du contenu multimodal ---
def extraire_contenu_pdf(pdf_path: Path, output_dir: Path = OUTPUT_DIR) -> Tuple[int, int, str]:
    """
    Extrait le texte page par page et toutes les images incorporées dans le PDF.
    Sauvegarde le texte dans output_dir/textes et les images dans output_dir/images.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    text_dir = output_dir / "textes"
    image_dir = output_dir / "images"
    text_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    textes = []
    nombre_images = 0

    with pymupdf.open(pdf_path) as doc:
        nombre_pages = len(doc)
        for num_page, page in enumerate(doc, start=1):
            texte = page.get_text("text", sort=True)
            fichier_texte = text_dir / f"page_{num_page:03d}.txt"
            fichier_texte.write_text(texte, encoding="utf-8")
            textes.append(f"===== PAGE {num_page} =====\n\n{texte}")

            images_vues = set()
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in images_vues:
                    continue
                images_vues.add(xref)
                try:
                    img_dict = doc.extract_image(xref)
                    if img_dict:
                        ext = img_dict["ext"]
                        fichier_img = image_dir / f"page_{num_page:03d}_image_{xref}.{ext}"
                        fichier_img.write_bytes(img_dict["image"])
                        nombre_images += 1
                except Exception as e:
                    print(f"Erreur extraction image page {num_page}, xref {xref} : {e}")

    texte_complet_path = output_dir / "texte_complet.txt"
    texte_complet_path.write_text("\n\n".join(textes), encoding="utf-8")

    doc_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:24]
    return nombre_pages, nombre_images, doc_id

# --- Étape 2 : Modèle CLIP & Embeddings Normalisés ---
class CLIPEmbedder:
    _instance = None

    def __init__(self, model_name=CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.bpe = SimpleTokenizer()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CLIPEmbedder()
        return cls._instance

    def decouper_texte(self, texte: str, taille: int = 60, chevauchement: int = 10):
        tokens = self.bpe.encode(texte)
        for debut in range(0, len(tokens), taille - chevauchement):
            fragment = self.bpe.decode(tokens[debut:debut + taille]).strip()
            if fragment:
                yield fragment
            if debut + taille >= len(tokens):
                break

    @torch.inference_mode()
    def encoder_textes(self, textes: List[str]) -> List[List[float]]:
        tokens = self.tokenizer(textes).to(self.device)
        vecteurs = self.model.encode_text(tokens).float()
        vecteurs = vecteurs / vecteurs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vecteurs.cpu().tolist()

    @torch.inference_mode()
    def encoder_image(self, chemin_image: Path) -> List[float]:
        chemin = Path(chemin_image)
        with Image.open(chemin) as img:
            tenseur = self.preprocess(img.convert("RGB"))
        tenseur = tenseur.unsqueeze(0).to(self.device)
        vecteur = self.model.encode_image(tenseur).float()
        vecteur = vecteur / vecteur.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vecteur.cpu().tolist()[0]

# --- Étape 3 : Base vectorielle ChromaDB ---
class MultimodalVectorStore:
    def __init__(self, chroma_dir: Path = CHROMA_DIR, collection_name: str = "pdf_clip_vit_b32"):
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = CLIPEmbedder.get_instance()

    def indexer_pdf(self, pdf_path: Path, output_dir: Path = OUTPUT_DIR) -> Tuple[int, int]:
        pdf_path = Path(pdf_path)
        doc_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:24]

        # Supprimer ancienne indexation du même document
        self.collection.delete(where={"document_id": doc_id})

        text_dir = output_dir / "textes"
        image_dir = output_dir / "images"

        ids = []
        documents = []
        metadatas = []

        # Indexer les textes
        for fichier in sorted(text_dir.glob("page_*.txt")):
            texte = fichier.read_text(encoding="utf-8")
            match = re.search(r"page_(\d+)", fichier.stem)
            page = int(match.group(1)) if match else -1

            for index, fragment in enumerate(self.embedder.decouper_texte(texte)):
                ids.append(f"{doc_id}:texte:{page}:{index}")
                documents.append(fragment)
                metadatas.append({
                    "type": "texte",
                    "document_id": doc_id,
                    "pdf": pdf_path.name,
                    "page": page,
                    "fragment": index,
                    "source": str(fichier.resolve())
                })

        batch_size = 64
        for i in range(0, len(documents), batch_size):
            lot_docs = documents[i:i + batch_size]
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=lot_docs,
                metadatas=metadatas[i:i + batch_size],
                embeddings=self.embedder.encoder_textes(lot_docs)
            )

        # Indexer les images
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jp2"}
        nb_images = 0
        for fichier in sorted(image_dir.iterdir()):
            if not fichier.is_file() or fichier.suffix.lower() not in extensions:
                continue
            vecteur = self.embedder.encoder_image(fichier)
            match = re.search(r"page_(\d+)", fichier.stem)
            page = int(match.group(1)) if match else -1

            self.collection.upsert(
                ids=[f"{doc_id}:image:{fichier.name}"],
                embeddings=[vecteur],
                documents=[f"Image extraite du PDF : {fichier.name}"],
                metadatas=[{
                    "type": "image",
                    "document_id": doc_id,
                    "pdf": pdf_path.name,
                    "page": page,
                    "source": str(fichier.resolve())
                }]
            )
            nb_images += 1

        return len(documents), nb_images

    # --- Étape 4 : Recherche multimodale par similarité cosinus ---
    def rechercher(
        self,
        texte: Optional[str] = None,
        image: Optional[Path] = None,
        type_resultat: Optional[Literal["texte", "image"]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Recherche par similarité cosinus acceptant texte ou image comme requête.
        Renvoie les k éléments les plus similaires.
        """
        if (texte is None) == (image is None):
            raise ValueError("Fournir soit un texte, soit une image.")
        if type_resultat not in (None, "texte", "image"):
            raise ValueError("type_resultat doit être 'texte', 'image' ou None.")

        total = self.collection.count()
        if total == 0:
            raise ValueError("La collection est vide.")

        if texte is not None:
            vecteur = self.embedder.encoder_textes([texte])[0]
        else:
            vecteur = self.embedder.encoder_image(image)

        options = {}
        if type_resultat is not None:
            options["where"] = {"type": type_resultat}

        return self.collection.query(
            query_embeddings=[vecteur],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
            **options
        )

# --- Étape 5 : Outil Agentique (Tool) ---
_store_global = None
def get_global_store() -> MultimodalVectorStore:
    global _store_global
    if _store_global is None:
        _store_global = MultimodalVectorStore()
    return _store_global

@tool
def retriever_tool(
    question: str,
    type_resultat: Literal["tous", "texte", "image"] = "tous",
    k: int = 6
) -> str:
    """
    Consulter la base vectorielle multimodale du document PDF 'Attention Is All You Need'.
    Permet d'extraire des explications textuelles, équations, résultats empiriques et figures/diagrammes d'architecture.

    Paramètres :
    - question : Requête en texte clair décrivant le concept, mécanisme ou figure recherché.
    - type_resultat : "tous" (texte et images), "texte" (passages textuels seuls), "image" (figures/diagrammes seuls).
    - k : Nombre de résultats les plus proches à retourner (1 à 10).
    """
    filtre = None if type_resultat == "tous" else type_resultat
    k_ajuste = max(1, min(k, 10))
    store = get_global_store()

    try:
        resultats = store.rechercher(texte=question, type_resultat=filtre, k=k_ajuste)
    except Exception as e:
        return json.dumps({"erreur": str(e)}, ensure_ascii=False)

    sources = []
    for idx, (doc_id, doc, meta, dist) in enumerate(
        zip(resultats["ids"][0], resultats["documents"][0], resultats["metadatas"][0], resultats["distances"][0]),
        start=1
    ):
        similarity = round(1.0 - dist, 3)
        sources.append({
            "reference": f"S{idx}",
            "id": doc_id,
            "type": meta["type"],
            "pdf": meta["pdf"],
            "page": meta["page"],
            "similarite": similarity,
            "source_path": meta["source"],
            "extrait": doc if meta["type"] == "texte" else f"Image file: {meta['source']}"
        })

    return json.dumps({
        "question": question,
        "nombre_sources": len(sources),
        "sources": sources
    }, ensure_ascii=False)

# --- Étape 6 : Construction de l'Agent IA (Pattern ReAct & LangGraph) ---
SYSTEM_PROMPT_AGENT = """Tu es un assistant scientifique expert en Deep Learning et architectures Transformer.
Tu as accès à l'outil `retriever_tool` pour consulter le papier fondateur 'Attention Is All You Need'.

Méthodologie ReAct :
1. Raisonnement (Thought) : Analyse la question de l'utilisateur et détermine les éléments textuels ou visuels (figures) nécessaires.
2. Action : Appelle `retriever_tool` avec des mots-clés précis (en anglais de préférence pour la recherche CLIP) et choisis `type_resultat` ("texte", "image", ou "tous").
3. Observation : Examine minutieusement les extraits et références retournés par l'outil.
4. Synthèse finale : Réponds en français de manière claire, structurée et technique, en citant systématiquement tes sources sous la forme [Attention Is All You Need, p. X]. Ne formule aucune affirmation qui n'apparaisse pas dans les sources.
"""

def creer_agent_langgraph(model=None):
    """
    Construit et compile le graphe d'exécution de l'agent ReAct avec LangGraph.
    Architecture MIND (LLM) et BODY (ToolNode).
    """
    tools = [retriever_tool]
    body_node = ToolNode(tools)

    # Si aucun modèle fourni, tenter ChatOpenAI ou fallback
    if model is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and api_key != "....":
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            # Fallback intelligent local pour les tests et exécutions sans clé externe
            class FallbackAgentLLM:
                def invoke(self, messages):
                    last_msg = messages[-1]
                    # Si c'est le message utilisateur initial
                    if isinstance(last_msg, HumanMessage) or (isinstance(last_msg, dict) and last_msg.get("role") == "user"):
                        q_text = last_msg.content if isinstance(last_msg, HumanMessage) else last_msg.get("content", "")
                        # Décision d'action ReAct
                        type_res = "image" if any(w in q_text.lower() for w in ["figure", "image", "diagramme", "schéma"]) else "tous"
                        return AIMessage(
                            content=f"Je dois interroger la base documentaire pour trouver les passages et figures pertinents.",
                            tool_calls=[{
                                "name": "retriever_tool",
                                "args": {"question": q_text, "type_resultat": type_res, "k": 4},
                                "id": "call_retriever_1"
                            }]
                        )
                    # Si c'est l'observation de l'outil
                    elif isinstance(last_msg, ToolMessage):
                        try:
                            data = json.loads(last_msg.content)
                            sources = data.get("sources", [])
                            res_text = "D'après le document *Attention Is All You Need* :\n\n"
                            for s in sources:
                                if s["type"] == "texte":
                                    res_text += f"- **Extrait (p. {s['page']})** : {s['extrait'][:180]}... [Attention is all you need.pdf, p. {s['page']}]\n"
                                else:
                                    res_text += f"- **Figure (p. {s['page']})** : {Path(s['source_path']).name} [Attention is all you need.pdf, p. {s['page']}]\n"
                            return AIMessage(content=res_text)
                        except Exception:
                            return AIMessage(content="Information récupérée avec succès depuis les sources du document.")
                    return AIMessage(content="Terminé.")
            model = FallbackAgentLLM()

    # Définition du graphe d'états
    workflow = StateGraph(MessagesState)

    def mind_node(state: MessagesState):
        messages = state["messages"]
        # Préfixer avec le system prompt si nécessaire
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT_AGENT)] + messages
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow.add_node("mind", mind_node)
    workflow.add_node("body", body_node)

    workflow.add_edge(START, "mind")
    workflow.add_conditional_edges("mind", tools_condition, {"tools": "body", END: END})
    workflow.add_edge("body", "mind")

    return workflow.compile()

# --- Étape 7 : Fonction d'interrogation et de test multimodal ---
def interroger_rag_agentique(
    agent,
    question: str,
    image_requete: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exécute le cycle ReAct de l'agent et extrait les réponses textuelles,
    les images retrouvées et la trace d'exécution.
    """
    contenu_requete = question
    if image_requete:
        contenu_requete += f" [Requête basée sur l'image : {Path(image_requete).name}]"

    initial_state = {
        "messages": [HumanMessage(content=contenu_requete)]
    }

    execution = agent.invoke(initial_state)

    messages = execution["messages"]
    reponse_finale = messages[-1].content
    appels_outils = []
    images_retrouvees = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            appels_outils.extend(msg.tool_calls)
        elif isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                for src in data.get("sources", []):
                    if src.get("type") == "image":
                        path_img = Path(src.get("source_path"))
                        if path_img.is_file() and path_img not in [img["path"] for img in images_retrouvees]:
                            images_retrouvees.append({
                                "path": path_img,
                                "page": src.get("page"),
                                "similarite": src.get("similarite")
                            })
            except Exception:
                pass

    if verbose:
        print("\n" + "="*50)
        print("QUESTION :", question)
        if image_requete:
            print("IMAGE ENTRÉE :", image_requete)
        print("="*50)
        print("\n--- RAISONNEMENT & APPELS D'OUTILS ---")
        for appel in appels_outils:
            print(f"Action : {appel.get('name')} | Args : {json.dumps(appel.get('args', {}), ensure_ascii=False)}")
        print("\n--- RÉPONSE FINALE ---")
        print(reponse_finale)
        if images_retrouvees:
            print(f"\n--- IMAGES MULTIMODALES ASSOCIÉES ({len(images_retrouvees)}) ---")
            for im in images_retrouvees:
                print(f"- Image page {im['page']} (Sim: {im['similarite']}) : {im['path'].name}")

    return {
        "question": question,
        "reponse": reponse_finale,
        "appels_outils": appels_outils,
        "images": images_retrouvees,
        "execution": execution
    }

