from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS  OLD use Pinecone for persistence & scalability
# NEW (Pinecone)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
# NEW (Custom Prompt)
from langchain_core.prompts import PromptTemplate
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


# Step 3 - Embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Step 4 - Vector DB
# OLD FAISS FAISS is local & not persistent
# vector_store = FAISS.from_documents(chunks, embeddings)
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
# NEW Pinecone (cloud, persistent, production-ready)
print("API KEY:", api_key)
index_name = "pdf-rag"

# connect to index
index = pc.Index(index_name)
# IMPORTANT: avoid duplicate uploads
vector_store = None  # 👈 ensures editor knows it exists

upload = input("Upload documents? (y/n): ")

if upload.lower() in ["y", "yes"]:
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas)

else:
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )


# Step 5 - Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# Step 6 - LLM (Ollama)
# llm = Ollama(model="llama3.2")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 7 - Custom Prompt
prompt_template = """
You are an AI assistant. Use ONLY the context below to answer.

If the answer is not in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# Step 8 - QA Chain

# ❌ OLD (default prompt) - commented because it is weak and may hallucinate
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever
# )

# ✅ NEW (custom prompt for better accuracy)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # helpful for debugging
    chain_type_kwargs={"prompt": PROMPT}
)


# =========================
# Step 9 - Chat Loop
# =========================
print("\n✅ Ready! Ask questions about your PDF.")
print("Type 'quit' to exit\n")

while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break

    result = qa_chain.invoke(question)

    print(f"\nAnswer: {result['result']}\n")

    # ✅ Show sources (optional but very useful)
    print("Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata}")

    print("-" * 50)