# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from argparse import ArgumentParser


class CLICommand(ABC):
    """
    Base class for command line tool.

    """

    @staticmethod
    @abstractmethod
    def define_args(parsers: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
