import argparse

from arf import __version__
from arf.constants import ENV_VARIABLES
from arf.models.trainer import train_resnet

""" TODO: Create a CLI for the ARF project. This cli should be able to:
 - Launch training experiments (for this three env variables should be defined:
    Training, validation and test). The model that will be trained will given
    by the user (resnet, inception or transformer).
 - Show statistics about the trainings and validations (to define which charts).
"""


def show_env_vars(*_) -> None:
    """
    Show the enviorment variables and its descriptions (from constants).
    """
    string_result = ""
    for var, var_description in ENV_VARIABLES.items():
        string_result += f" - {var:20s}: {var_description}\n"
    print(string_result)


PARSER = argparse.ArgumentParser(description="""
    This is the app for the ARF project. This app is based on test how good and
    easy is the development on PyTorch. For that, three deep learning networks
    are used: ResNet, Inception and Transformer. This networks will be trained
    and tested using PyTorch and related third party libraries. For the
    configuration, a .env file is used.
""")

PARSER.add_argument('--version', action='version', version=__version__)
PARSER.add_argument('-ed', '--env-definition', action=argparse.BooleanOptionalAction,
                    help='Show the env variables and its descriptions.')
PARSER.add_argument('--train-resnet', action=argparse.BooleanOptionalAction,
                    help='Train resnet network.')

ARGUMENTS_MAPPING = {
    "env_definition": show_env_vars,
    "train_resnet": train_resnet
}


def main() -> None:
    command_line_arguments = vars(PARSER.parse_args())
    for argument, argument_function in ARGUMENTS_MAPPING.items():
        if argument in command_line_arguments and command_line_arguments[argument]:
            ARGUMENTS_MAPPING[argument]()
