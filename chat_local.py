from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import sys

# 1. On se connecte DIRECTEMENT au dossier existant (sans charger les PDF)
print("Branchement au cerveau local (ma_base_indexee)...")
try:
    vectorstore = Chroma(
        persist_directory="./ma_base_indexee", 
        embedding_function=OllamaEmbeddings(model="llama3.2")
    )
except Exception as e:
    print("❌ Erreur : La base n'existe pas encore. Lance d'abord ton premier script d'ingestion.")
    sys.exit()

# 2. On allume le moteur Llama 3.2
llm = OllamaLLM(model="llama3.2", temperature=0)

def interroger_la_memoire(question):
    # On fouille dans le dossier pour trouver les bons passages
    docs_pertinents = vectorstore.similarity_search(question, k=3)
    contexte = "\n\n".join([doc.page_content for doc in docs_pertinents])
    
    # On envoie les passages trouvés + la question à l'IA
    prompt = f"Tu es un expert souverain. Utilise UNIQUEMENT ce contexte pour répondre : {contexte}\n\nQuestion : {question}"
    return llm.invoke(prompt)

# 3. La boucle de discussion infinie
print("\n🤖 IA prête et connectée à tes documents !")
print("(Tape 'quit' pour quitter le logiciel)\n")

while True:
    question_utilisateur = input("👉 Ta question : ")
    
    if question_utilisateur.lower() == 'quit':
        print("Fermeture du système souverain. À bientôt !")
        break
        
    print("⏳ Recherche dans les documents et réflexion...")
    reponse = interroger_la_memoire(question_utilisateur)
    
    print("\n🤖 RÉPONSE :")
    print(reponse)
    print("-" * 50)