import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# RecursiveCharacterTextSplitter a été déplacé selon les versions de LangChain :
# essaie d'abord le nouveau package dédié, sinon retombe sur l'ancien chemin.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# PromptTemplate a aussi bougé selon les versions.
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate


# ====================================================================
# ÉTAPE 1 : Initialisation de l'environnement et squelette de l'interface
# ====================================================================
# Ici : l'état de session (mémoire de l'app entre les interactions).
# Le squelette graphique proprement dit (barre latérale + zone de
# conversation) est construit plus bas, une fois toutes les fonctions
# des étapes 2/3/4 définies (elles y sont appelées).

if "messages" not in st.session_state:
    st.session_state.messages = []  # historique de conversation

if "db" not in st.session_state:
    st.session_state.db = None  # base vectorielle Chroma (créée après indexation)

if "files" not in st.session_state:
    st.session_state.files = []  # liste des fichiers déjà indexés (pour affichage)


# ====================================================================
# ÉTAPE 2 : Le Pipeline d'Ingestion (Traitement des Données)
# ====================================================================

EMBED_MODEL = "all-MiniLM-L6-v2"  # modèle d'embeddings local (léger et rapide)
DB_DIR = "./chroma_db"            # dossier de persistance de la base vectorielle

CHUNK_SIZE = 800  # taille des segments de texte (en caractères)
OVERLAP = 150      # chevauchement entre segments (~18% du chunk_size)
# Justification : 800 caractères ~ 1 paragraphe, assez grand pour garder du
# contexte, assez petit pour rester précis lors de la recherche sémantique.
# L'overlap de 150 évite de couper une idée pile à la frontière de deux chunks.


@st.cache_resource
def get_embeddings():
    """Charge le modèle d'embeddings une seule fois (mis en cache par Streamlit)."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_document(uploaded_file):
    """
    Extraction : extrait le texte d'un fichier uploadé (PDF, TXT ou MD).
    Streamlit fournit un objet en mémoire : on l'écrit temporairement sur
    disque pour pouvoir utiliser les DocumentLoaders de LangChain.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    # Écriture temporaire du fichier sur disque
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            st.warning(f"Format non supporté : {uploaded_file.name}")
            return []

        docs = loader.load()

        # On force le nom de métadonnée "source" au nom original du fichier
        # (et non le chemin temporaire, qui n'a aucun sens pour l'utilisateur)
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        return docs

    finally:
        os.remove(path)  # nettoyage du fichier temporaire


def chunk_documents(docs):
    """Chunking : découpe les documents en segments (chunks) avec chevauchement."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # essaie de couper aux frontières naturelles
    )
    return splitter.split_documents(docs)


def index_files(uploaded_files):
    """
    Pipeline complet d'ingestion :
    Extraction -> Chunking -> Vectorisation -> Stockage dans Chroma.
    """
    docs = []

    for uploaded_file in uploaded_files:
        loaded = load_document(uploaded_file)
        docs.extend(loaded)

    if not docs:
        st.error("Aucun contenu exploitable n'a été extrait des fichiers.")
        return

    chunks = chunk_documents(docs)

    emb = get_embeddings()

    # Vectorisation : création (ou remplacement) de la base vectorielle
    # Chroma en local, métadonnées (source) conservées via les chunks.
    db = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=DB_DIR,
    )

    st.session_state.db = db
    st.session_state.files = [f.name for f in uploaded_files]

    st.success(f"{len(chunks)} segments indexés depuis {len(uploaded_files)} fichier(s).")


# ====================================================================
# ÉTAPE 3 : Mode "Recherche Sémantique" (Toggle Désactivé)
# ====================================================================
# Ce mode sert à auditer et valider le bon fonctionnement de la base
# vectorielle : aucun appel à un modèle génératif.

TOP_K = 4  # nombre de chunks récupérés à chaque requête


def semantic_search(query):
    """
    Interroge la base vectorielle et retourne les chunks les plus proches
    sémantiquement de la requête, SANS appel à un LLM.
    """
    return st.session_state.db.similarity_search(query, k=TOP_K)


def format_results(results):
    """Met en forme les résultats bruts de la recherche sémantique pour l'affichage."""
    if not results:
        return "Aucun résultat pertinent trouvé."

    output = ""
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "Source inconnue")
        output += f"**Extrait {i}** — *source : {source}*\n\n"
        output += f"> {doc.page_content.strip()}\n\n---\n\n"
    return output


# ====================================================================
# ÉTAPE 4 : Mode "RAG Complet" (Toggle Activé)
# ====================================================================
# C'est l'aboutissement de l'application : le LLM répond en se basant
# STRICTEMENT sur les fragments récupérés (comme à l'Étape 3).

LLM_MODEL = "mistral"  # modèle LLM local servi par Ollama

# Prompt système strict : on force le LLM à ne répondre qu'à partir du contexte fourni
PROMPT_TEMPLATE = """Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni ci-dessous.
Règles impératives :
- Utilise uniquement les informations présentes dans le contexte.
- Si la réponse ne se trouve pas dans le contexte, réponds : "Je ne trouve pas cette information dans les documents fournis."
- Ne fais aucune supposition et n'invente aucune information extérieure au contexte.
- Réponds en français, de façon claire et concise.

Contexte :
{context}

Question :
{question}

Réponse :"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


@st.cache_resource
def get_llm():
    """Charge le client Ollama une seule fois (mis en cache par Streamlit)."""
    return Ollama(model=LLM_MODEL)


def rag_answer(query):
    """
    Pipeline RAG complet :
    1. Récupération des chunks pertinents (Étape 3)
    2. Construction du prompt avec le contexte (ingénierie de prompt)
    3. Appel au LLM local (Ollama)
    Retourne la réponse du LLM ainsi que les documents sources utilisés.
    """
    docs = st.session_state.db.similarity_search(query, k=TOP_K)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT.format(context=context, question=query)

    llm = get_llm()
    answer = llm.invoke(prompt)

    return answer, docs


# ====================================================================
# ÉTAPE 1 (suite) : Squelette de l'interface — Barre latérale
# ====================================================================
# Zone de téléchargement de fichiers, bouton d'indexation (câblé à la
# fonction index_files de l'Étape 2) et toggle d'activation du LLM
# (bascule entre les modes des Étapes 3 et 4).

with st.sidebar:
    st.title("📚 Configuration")

    st.markdown("### 1. Charger des documents")
    uploaded_files = st.file_uploader(
        "Formats acceptés : PDF, TXT, MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button(" Indexer les documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Indexation en cours..."):
                index_files(uploaded_files)
        else:
            st.warning("Merci de charger au moins un fichier avant d'indexer.")

    if st.session_state.files:
        st.markdown("**Fichiers indexés :**")
        for name in st.session_state.files:
            st.markdown(f"- {name}")

    st.divider()

    st.markdown("### 2. Mode de fonctionnement")
    use_llm = st.toggle("Activer le LLM (mode RAG complet)", value=False)

    if use_llm:
        st.caption("🤖 Mode Assistant RAG complet — génération via Ollama")
    else:
        st.caption("🔍 Mode Recherche Sémantique pure — pas de génération LLM")


# ====================================================================
# ÉTAPE 1 (suite) : Squelette de l'interface — Zone principale (conversation)
# ====================================================================
# Interface conversationnelle avec historique. Selon la position du
# toggle, la question de l'utilisateur déclenche le mode Étape 3
# (recherche sémantique pure) ou le mode Étape 4 (RAG complet).

def show_sources(sources):
    """Affiche une liste de documents sources dans un expander."""
    with st.expander("📎 Extraits utilisés comme contexte"):
        for i, doc in enumerate(sources, start=1):
            source = doc.metadata.get("source", "Source inconnue")
            st.markdown(f"**Extrait {i}** — *source : {source}*")
            st.markdown(f"> {doc.page_content.strip()}")


st.title("🗂️ NotebookLM Local")
st.caption("Système RAG 100% local — vos documents ne quittent jamais votre machine.")

# Affichage de l'historique de conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si un message assistant possède des sources, on les affiche dans un expander
        if message.get("sources"):
            show_sources(message["sources"])

# Zone de saisie utilisateur
query = st.chat_input("Posez une question sur vos documents...")

if query:
    # Vérification qu'une base vectorielle existe avant toute requête
    if st.session_state.db is None:
        st.error("Veuillez d'abord charger et indexer des documents.")
    else:
        # Affichage du message utilisateur
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Traitement selon le mode sélectionné
        with st.chat_message("assistant"):
            if use_llm:
                # --- Mode RAG complet (Étape 4) ---
                with st.spinner("Génération de la réponse..."):
                    answer, sources = rag_answer(query)

                st.markdown(answer)
                show_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            else:
                # --- Mode Recherche Sémantique pure (Étape 3) ---
                with st.spinner("Recherche en cours..."):
                    results = semantic_search(query)

                formatted = format_results(results)
                st.markdown(formatted)

                st.session_state.messages.append(
                    {"role": "assistant", "content": formatted, "sources": None}
                )
