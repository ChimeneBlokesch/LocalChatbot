# PersonalChatbot
Personal project to learn how to work with the LangChain package. The user can ask questions about their own private documents with a miniLM (default: Ollama Llama3.1). The model can be run locally, therefore not requiring external computational resource.

# How to Install
Copy this repository and use the `environment.yml` or `requirements.txt` files to install the Python packages. The code is developed with Python 3.10.6 on Windows 11.

Currently, Ollama Llama3.1 is used as the Language Model (LM). This requires Ollama desktop app to be installed. After completion, download the model using `ollama pull llama3.1` in the terminal.

# How to Use
1. Create a `data` directory in the same folder as this README.
2. Copy your own files (.pdf, .txt, .docx, .tex) to the newly created `data` folder. Use subfolders to group the files. These groups will be called datasets.
3. Run `python personalChatbot.py --dataset DATASET`, where `DATASET` is a placeholder for the name of the subfolder.
4. Repeatedly ask questions (queries) about the documents in the `data/DATASET` folder.
5. [Optional] Check the automatically created `logs/DATASET` folder for the documents which are the most similar to the query. This is the context used to answer the question.
