import streamlit as st
import time

# On essaie d'importer les modules un par un pour détecter l'erreur
try:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
except ImportError as e:
    st.error(f"Il manque une bibliothèque : {e}")
    st.info("Tape : pip install langchain-chroma langchain-ollama")
    st.stop()

st.title("🤖 Mon IA sur Mobile")

# Chargement du cerveau avec indicateur de progrès
with st.status("Initialisation de Llama 3.2...", expanded=True) as status:
    try:
        st.write("Connexion à la base de données...")
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = Chroma(persist_directory="./ma_base_indexee", embedding_function=embeddings)
        
        st.write("Réveil de Llama 3.2 (Ollama)...")
        llm = OllamaLLM(model="llama3.2", timeout=30)
        
        status.update(label="✅ Système prêt !", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Erreur au démarrage : {e}")
        st.stop()

# Interface de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose ta question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("L'IA réfléchit..."):
            docs = vectorstore.similarity_search(prompt, k=3)
            contexte = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"Réponds en français. Contexte : {contexte}\n\nQuestion : {prompt}"
            
            try:
                response = llm.invoke(full_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"L'IA n'a pas répondu : {e}")