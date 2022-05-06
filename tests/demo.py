import timeit

from pybasic import basic, correct_illumination
from pybasic.tools import load_data
from pybasictorch.basic import correct_illumination as ci_torch
from pybasictorch.utils import load_data as load_data_torch


def with_original():
    brain = load_data("../ExampleData/WSI_brain", verbosity=False)
    correct_illumination(brain, *basic(brain, darkfield=False, verbosity=False))


def with_torch():
    corrected = ci_torch(load_data_torch("../ExampleData/WSI_brain").cuda())
    corrected.cpu()


if __name__ == "__main__":
    print(timeit.timeit("with_original()", number=2, setup="from __main__ import with_torch, with_original"))
    print(timeit.timeit("with_torch()", number=2, setup="from __main__ import with_torch, with_original"))

