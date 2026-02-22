from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import os
import sys

# 1. On récupère le chemin du dossier où se trouve le script
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(current_dir, "mes_documents")

print(f"Recherche des documents dans : {docs_path}")

# --- NOUVEAUTÉ : SÉCURITÉ ANTI-CRASH ---
if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    print(f"\n✅ Dossier 'mes_documents' créé automatiquement à : {docs_path}")
    print("⚠️ ATTENTION : Le dossier est vide !")
    print("👉 Action requise : Glisse au moins un vrai fichier .pdf dans ce dossier, puis relance le script.")
    sys.exit() # On arrête le programme proprement
# ---------------------------------------

# 2. Charger tous les PDF d'un dossier
print("\nAnalyse des documents en cours...")
loader = DirectoryLoader(docs_path, glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# --- SÉCURITÉ 2 : Vérifier qu'il y a bien des PDF ---
if not docs:
    print("❌ Erreur : Le dossier existe, mais aucun fichier PDF n'a été trouvé à l'intérieur.")
    sys.exit()

print(f"📖 {len(docs)} pages trouvées. Découpage en cours...")

# ... (Laisse la suite de ton code intacte à partir d'ici : text_splitter, Chroma, etc.)

# 2. Découper les textes en morceaux gérables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Créer la base de données vectorielle (locale et souveraine)
# C'est ici qu'on transforme le texte en mathématiques
print("Création de la mémoire vectorielle...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OllamaEmbeddings(model="llama3.2"),
    persist_directory="./ma_base_indexee" # Le dossier où sera stockée la mémoire
)

# 4. Fonction de recherche et réponse
def interroger_la_memoire(question):
    # Trouver les 3 passages les plus pertinents
    docs_pertinents = vectorstore.similarity_search(question, k=10)
    contexte = "\n\n".join([doc.page_content for doc in docs_pertinents])
    
    llm = OllamaLLM(model="llama3.2")
    prompt = f"""Tu es un assistant IA strictement factuel. 
    Voici plusieurs extraits d'un document :
    {contexte}
    
    Consignes :
    1. Trouve la réponse à la question en utilisant uniquement ces extraits.
    2. Ignore totalement les extraits qui ne sont pas pertinents pour la question.
    3. Ne fais aucun commentaire sur le contexte, donne juste la réponse directement.
    4. Si la réponse n'est pas dans les extraits, dis simplement "Je ne sais pas".
    
    Question : {question}
    """
    return llm.invoke(prompt)

# Test
print("\n🤖 Prêt ! Posez votre question.")
print(interroger_la_memoire("Quelles sont les clauses de confidentialité du contrat ?"))