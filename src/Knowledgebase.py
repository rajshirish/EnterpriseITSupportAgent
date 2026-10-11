from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

knowledge_base_dir = Path(__file__).resolve().parent.parent / "KnowledgeBase"
pdf_files = sorted(knowledge_base_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDF documents found in {knowledge_base_dir}")

pdf_pages = [
    page
    for pdf_file in pdf_files
    for page in PyPDFLoader(str(pdf_file)).load()
]
print(f"Loaded {len(pdf_files)} documents with {len(pdf_pages)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ",", " "]
)
split_docs = splitter.split_documents(pdf_pages)
print(f"Total number of chunks: {len(split_docs)}")

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(split_docs, embedding=embedding_model)
vector_store.save_local("Enterprise_IT_knowledge_base")
print("Vector store created using FAISS")