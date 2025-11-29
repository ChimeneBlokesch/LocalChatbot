from langchain_community.document_loaders import (PyPDFLoader,
                                                  Docx2txtLoader,
                                                  TextLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document


def has_extension(file: str, extensions: list[str]):
    for ext in extensions:
        if file.endswith(ext):
            return True

    return False


def load_file(file: str) -> list[Document]:
    """
    Use document loaders to read the file as document. Also splits the file
    into multiple documents, depending on the chunk size.
    """
    loader = None
    splitter = None

    if has_extension(file, [".txt", ".tex", ".md"]):
        loader = TextLoader(file, encoding='UTF-8')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)

    if file.endswith(".docx"):
        loader = Docx2txtLoader(file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)

    if loader is None:
        print("Warning: the file type of ", file, "is not supported")
        return []

    docs = loader.load()

    if splitter is not None:
        # Split the file into multiple document chunks
        docs = splitter.split_documents(docs)

    return docs
