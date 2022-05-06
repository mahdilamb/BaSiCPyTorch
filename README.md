# PyBaSiC (TorchVision)

A version of the flat-field correction algorithm developed in
the [Marr/Peng lab](http://www.nature.com/articles/ncomms14836), modified to work with TorchVision and therefore can be
moved to the GPU.

Important changes to implementation:

- Computations are performed with single rather than double-precision
- The up/downscaling functions are done with the torchvision Rescale function which is not exactly the same as the
  skimage implementation

The original version can be found [here](https://github.com/peng-lab/BaSiCPy)

All credit for the original algorithm goes to the original authors