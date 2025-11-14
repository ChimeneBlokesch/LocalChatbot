# , UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file: str) -> list[Document]:
    loader = None
    splitter = None

    if file.endswith(".tex") or file.endswith(".txt"):
        loader = TextLoader(file, encoding='UTF-8')

    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)

    if file.endswith(".docx"):
        loader = Docx2txtLoader(file)
        # loader = UnstructuredWordDocumentLoader(file, mode="elements")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)

    if loader is None:
        print("Warning: the file type of ", file, "is not supported")
        return []

    docs = loader.load()

    if splitter is not None:
        docs = splitter.split_documents(docs)

    return docs
