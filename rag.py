
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.prompts.prompt import PromptTemplate

import os

from argument_parser import get_args
from personalChatbot import PersonalChatbot

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


class RAGChatbot(PersonalChatbot):

    def query(self, query: str) -> tuple[str, str]:
        """
        Input a prompt to the LM, where the most similar documents
        are also given to the LM as context.
        """
        db = Chroma(persist_directory=self.chroma_folder,
                    embedding_function=self.embeddings)

        results = db.similarity_search_with_relevance_scores(query, k=50)

        if len(results) == 0:
            print("Unable to find matching results.")

        self.log_results(results)

        # Combine the matching documents to obtain the context
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
        """
        Write the similar documents and their similarity score to a file
        in the logs folder. Uses the query id as the filename.
        """
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

            f.write("The best result has a similarity score of "
                    f"{best_score:2f}%")


if __name__ == "__main__":
    arg_parser = get_args()

    chatbot = RAGChatbot(dataset=arg_parser.dataset,
                         documents_folder=arg_parser.documents_folder,
                         log_folder=arg_parser.log_folder)

    while True:
        query = input("New query:")
        metadata, response = chatbot.query(query)
        print(response)
