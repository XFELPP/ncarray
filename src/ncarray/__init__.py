import os

from ncarray.core._pyncarray import *


def get_include() -> str:
    return os.path.join(os.path.dirname(__file__), "include")


def get_lib_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "lib")
