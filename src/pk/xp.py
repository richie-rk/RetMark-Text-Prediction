import functools

import numpy

live = numpy


@functools.cache
def consolidation_backend():
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy, "cuda"
    # A missing or broken CUDA install surfaces as many different exception
    # types; any of them means "no usable GPU", never a hard failure.
    except Exception:
        pass
    return numpy, "cpu"
