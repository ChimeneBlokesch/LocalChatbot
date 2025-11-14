from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.prompts.prompt import PromptTemplate

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

import os

import utils
from argument_parser import get_args

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_CHROMA_FOLDER = "./local_vectorstore"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


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
        llm = Ollama(
            model=self.llm_model_name,
            request_timeout=360.0,
            context_window=1024,
        )
        Settings.llm = llm

        return llm

    def load_docs(self) -> list[Document]:
        docs = []
        for file in os.listdir(self.docs_folder):
            new_docs = utils.load_file(os.path.join(self.docs_folder, file))
            docs += new_docs
            print(len(docs), len(new_docs))

        return docs

    def store_local_embeddings(self, docs: list[Document]) -> Chroma:
        vector_store = Chroma.from_documents(
            docs, self.embeddings, persist_directory=self.chroma_folder)

        return vector_store

    def query_rag(self, query: str) -> str:
        db = Chroma(persist_directory=self.chroma_folder,
                    embedding_function=self.embeddings)

        results = db.similarity_search_with_relevance_scores(query, k=50)

        if len(results) == 0:
            print("Unable to find matching results.")

        self.log_results(results)

        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join(
            [doc.page_content for doc, _score in results])

        # Create prompt template using context and query text
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

        formatted_prompt = prompt_template.format(context=context_text,
                                                  question=query)

        response_text = self.llm.complete(formatted_prompt).text

        # Get sources of the matching documents
        sources = [doc.metadata.get("source", None) for doc, _score in results]

        # Format and return response including generated text and sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        self.query_id += 1

        return formatted_response, response_text

    def log_results(self, results: list[tuple[Document, float]]) -> None:
        os.makedirs(self.log_folder, exist_ok=True)

        file = os.path.join(self.log_folder, f"{self.query_id}.txt")

        with open(file, "w", encoding="utf-8") as f:
            for i, (doc, score) in enumerate(results):
                f.write(f"Result {i} with score {score}\n")
                f.write(doc.page_content)
                f.write("\n")

            f.write("\n")

            # Percentage of the result with the highest score
            best_score = results[0][1] * 100

            f.write(
                f"The best result has a similarity score of {best_score:2f}%")


if __name__ == "__main__":
    arg_parser = get_args()

    chatbot = PersonalChatbot(dataset=arg_parser.dataset,
                              documents_folder=arg_parser.documents_folder,
                              log_folder=arg_parser.log_folder)

    while True:
        query = input("New query:")
        metadata, response = chatbot.query_rag(query)
        print(response)
