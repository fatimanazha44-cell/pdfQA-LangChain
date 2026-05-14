from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

from dotenv import load_dotenv
import os
load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Step 1 - Load PDF
pdf_path = input("Enter the path to your PDF file: ")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Step 2 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3 - Embed + store in FAISS
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 4 - Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 5 - Connect to Ollama
# llm = Ollama(model="llama3.2")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
# Step 6 - Build the chain (RetrievalQA connects retriever + llm)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("\n✅ Ready! Ask questions about your PDF.")
print("Type 'quit' to exit\n")

# Step 7 - Ask questions
while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break
    answer = qa_chain.invoke(question)
    print(f"\nAnswer: {answer['result']}\n")
    print("-" * 50)