import argparse


class Arguments:
    def __init__(self,
                 data_folder: str,
                 root_folder: str):
        self.data_folder = data_folder
        self.root_folder = root_folder


def get_args():
    parser = argparse.ArgumentParser(
        prog='LocalChatbot',
        description="Applying language models locally on private documents. ")

    parser.add_argument("--data_folder", required=True,
                        help="Folder of the document files "
                        "within the root folder")

    parser.add_argument("--root_folder", default="data",
                        help="Root folder of the data")

    args = parser.parse_args(namespace=Arguments)
    return args
