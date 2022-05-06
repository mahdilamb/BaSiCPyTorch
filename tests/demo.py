import torch
from tifffile import imsave

from pybasic import basic, correct_illumination
from pybasic.tools import load_data
from pybasictorch.basic import correct_illumination as ci_torch
from pybasictorch.utils import load_data as load_data_torch


def with_original():
    brain = load_data("../ExampleData/WSI_brain", verbosity=False)
    return correct_illumination(brain, *basic(brain, darkfield=False, verbosity=False))


def with_torch():
    corrected = ci_torch(load_data_torch("../ExampleData/WSI_brain").cuda(), in_place=True, precision=torch.float64)
    return corrected.cpu()


def with_torch_cpu():
    return ci_torch(load_data_torch("../ExampleData/WSI_brain"))


if __name__ == "__main__":
    imsave("corrected_og.tif", with_original())
    """number = 3
    print("original()")
    print(timeit.timeit("with_original()", number=number, setup="from __main__ import with_torch, with_original"))
    print("torch()")
    print(
        timeit.timeit("with_torch()", number=number, setup="from __main__ import with_torch, with_original"))
    print("with_torch_cpu()")
    print(timeit.timeit("with_torch_cpu()", number=number,
                        setup="from __main__ import with_torch, with_original, with_torch_cpu"))"""
