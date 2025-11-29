from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

import os

import utils
from argument_parser import get_args

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_CHROMA_FOLDER = "./local_vectorstore"


class PersonalChatbot:
    def __init__(self,
                 dataset: str,
                 log_folder: str = "log",
                 documents_folder: str = "data",
                 embeddings_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 llm_model_name: str = DEFAULT_LLM_MODEL,
                 chroma_folder: str = DEFAULT_CHROMA_FOLDER):
        self.embeddings_model_name = embeddings_model_name
        self.llm_model_name = llm_model_name

        self.dataset = dataset

        self.docs_folder = os.path.join(documents_folder, dataset)
        self.chroma_folder = os.path.join(chroma_folder, dataset)
        self.log_folder = os.path.join(log_folder, dataset)

        self.llm = self.setup_llm()

        docs = self.load_docs()

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model_name)

        self.vector_store = self.store_local_embeddings(docs)

        # Keep track of the number of queries for the logging
        self.query_id = 0

    def setup_llm(self) -> Ollama:
        """
        Create the language model using Ollama
        """
        llm = Ollama(
            model=self.llm_model_name,
            request_timeout=360.0,
            # Limit window to avoid having too low system memory
            # to load the model
            context_window=1024,
        )
        Settings.llm = llm

        return llm

    def load_file_as_documents(self, file: str) -> list[Document]:
        return utils.load_file(os.path.join(self.docs_folder, file))

    def load_docs(self) -> list[Document]:
        """
        Search the dataset folder for the input files
        """
        docs: list[Document] = []

        for file in os.listdir(self.docs_folder):
            new_docs = self.load_file_as_documents(file)
            docs += new_docs

        return docs

    def store_local_embeddings(self, docs: list[Document]) -> Chroma:
        """
        Convert the documents to local embeddings and store them in the
        Chroma folder.
        """
        vector_store = Chroma.from_documents(
            docs, self.embeddings, persist_directory=self.chroma_folder)

        return vector_store

    def query(self, query: str) -> tuple[str, str]:
        raise NotImplementedError()

    def log_results(self, results: list[tuple[Document, float]]) -> None:
        raise NotImplementedError()
