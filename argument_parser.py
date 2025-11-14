import argparse


class Arguments:
    def __init__(self,
                 dataset: str,
                 documents_folder: str,
                 log_folder: str):
        self.dataset = dataset
        self.documents_folder = documents_folder
        self.log_folder = log_folder


def get_args():
    parser = argparse.ArgumentParser(
        prog='LocalChatbot',
        description="Applying language models locally on private documents. ")

    parser.add_argument("--dataset", required=True,
                        help="Name of the documents group. "
                        "Must match with folder names in `documents_folder`")

    parser.add_argument("--documents_folder", default="data",
                        help="Root folder with the document files")

    parser.add_argument("--log_folder", default="log",
                        help="Root folder with the logging files")

    args = parser.parse_args(namespace=Arguments)
    return args
