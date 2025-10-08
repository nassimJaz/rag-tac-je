import streamlit as st
from app.generation.generate_response import GenerateResponse
from app.retriever.qdrant_retriever import QdrantRetriever
from cli_ans import extracts_sources
from app.ingestion.ingest import ingest
# Initialisation avec cache pour éviter les réinstanciations
@st.cache_resource(ttl=4000)
def get_retriever():
    return QdrantRetriever()

# Cache pour les résultats de recherche (valable 10 minutes)
@st.cache_data(ttl=4000, show_spinner="Recherche des documents pertinents...")
def retrieve_documents(_retriever, query):
    return _retriever.retrieve(query)

# Cache pour la génération de réponse (valable 5 minutes)
@st.cache_data(ttl=4000, show_spinner="Génération de la réponse...")
def generate_response(_docs, query):
    return GenerateResponse(_docs, query).generate()

def get_chatbot_response(query, _retriever, use_cache=True) -> str:
    try:
        if use_cache:
            # Récupération des documents avec cache
            docs_retrieved = retrieve_documents(_retriever, query)
            # Génération de la réponse avec cache
            output = generate_response(docs_retrieved, query)
        else:
            # Mode sans cache (pour debug)
            docs_retrieved = _retriever.retrieve(query)
            output = GenerateResponse(docs_retrieved, query).generate()
        
        # Optionnel: ajout des sources (décochez si trop lent)
        # output += extracts_sources(query, docs_retrieved)
        
        return output
    except Exception as e:
        raise Exception(f"Erreur lors de la génération de réponse: {str(e)}")

# Configuration de la page
st.set_page_config(page_title="Chatbot KiwiX JE", layout="wide")

# Initialisation du retriever (une seule fois)
retriever = get_retriever()

# INITIALISATION CRUCIALE DES SESSION_STATE EN PREMIER
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

col1, col2 = st.columns([4, 1])
with col1:
    st.title("Chatbot JE")
with col2:
    if st.button("Lancer l'ingestion"):
        with st.spinner("Ingestion en cours, soyez très patient svp ..."):
            try:
                ingest()
                st.success("Ingestion terminée avec succès!")
            except Exception as e:
                st.error(f"Erreur lors de l'ingestion: {e}")

# Sidebar avec paramètres de performance
with st.sidebar:
    st.header("⚙️ Paramètres de performance")
    
    # Option pour le cache
    use_cache = st.checkbox("Utiliser le cache", value=True, 
                           help="Décochez pour forcer le recalcul à chaque fois (débogage)")
    
    # Boutons de gestion du cache
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer le cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache effacé avec succès!")
            
    with col2:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()
    
    # Limite de l'historique des messages
    history_limit = st.slider("Limite messages historiques", 
                             min_value=10, max_value=100, value=20, step=5,
                             help="Nombre maximum de messages à conserver dans l'historique")
    
    # Statistiques de performance (MAINTENANT EN SÉCURITÉ)
    st.subheader("📊 Statistiques")
    st.write(f"Nombre de messages: {len(st.session_state.messages)}")
    
    if st.session_state.last_query:
        display_query = st.session_state.last_query[:50] + "..." if len(st.session_state.last_query) > 50 else st.session_state.last_query
        st.write(f"Dernière requête: {display_query}")
    else:
        st.write("Dernière requête: Aucune")
    
    # Information sur le cache
    if st.checkbox("ℹ️ Afficher les infos cache"):
        st.write("**Cache configuré:**")
        st.write("- Retriever: 1 heure")
        st.write("- Recherche: 10 minutes") 
        st.write("- Génération: 5 minutes")
    
    # Option pour les sources
    show_sources = st.checkbox("Afficher les sources", value=False,
                              help="Ajouter les sources des documents à la réponse")
    
    st.markdown("---")
    st.info("💡 **Conseil:** Utilisez le cache pour de meilleures performances en production.")

# Nettoyer l'historique si trop long
if len(st.session_state.messages) > history_limit * 2:  # Marge de sécurité
    st.session_state.messages = st.session_state.messages[-history_limit:]
    st.sidebar.warning(f"Historique réduit à {history_limit} messages")

# Afficher les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Champ de saisie pour l'utilisateur
user_input = st.chat_input("Posez une question ...")

# Vérifier si la requête est nouvelle pour éviter les traitements inutiles
if user_input and user_input != st.session_state.last_query:
    st.session_state.last_query = user_input
    
    try:
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Afficher immédiatement le message utilisateur
        with st.chat_message("user"):
            st.markdown(user_input)

        # Créer un placeholder pour la réponse
        response_placeholder = st.empty()
        
        with st.chat_message("assistant"):
            # Obtenir la réponse du chatbot avec gestion visuelle
            with st.spinner("🔍 Recherche des documents pertinents..."):
                chatbot_response = get_chatbot_response(user_input, retriever, use_cache)
            
            # Ajouter les sources si demandé
            if show_sources:
                try:
                    docs_retrieved = retriever.retrieve(user_input)
                    sources = extracts_sources(user_input, docs_retrieved)
                    chatbot_response += sources
                except Exception as e:
                    chatbot_response += f"\n\n*⚠️ Impossible d'ajouter les sources: {str(e)}*"
            
            # Afficher la réponse
            response_placeholder.markdown(chatbot_response)
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
        
    except Exception as e:
        # Gestion d'erreur plus propre
        error_message = f"❌ Erreur: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# Bouton pour effacer l'historique dans l'interface principale
if st.session_state.messages:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🧹 Effacer toute la conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_query = ""
            st.rerun()