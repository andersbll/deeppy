.. _installation-guide:

Installation guide
==================

DeepPy relies on `CUDArray <http://github.com/andersbll/cudarray>`_ for most of
its calculations. Therefore, you must first `install CUDArray
<http://github.com/andersbll/cudarray#installation>`_. Note that you can choose
to install CUDArray `without the CUDA back-end
<http://github.com/andersbll/cudarray#without-cuda-back-end>`_ which simplifies
the installation process.

With CUDArray installed, you can install DeepPy with the standard::

    git clone git@github.com:andersbll/deeppy.git
    cd deeppy
    python setup.py install

If you wish to extend/modify/debug DeepPy for your own project, you should
consider the :code:`develop` installation instead::

    python setup.py develop


Verify CUDA back-end
--------------------
If CUDArray's CUDA back-end fails to start, CUDArray will automatically
fallback to its NumPy/Cython back-end. This feature can make it difficult to
determine if the GPU is actually being used. To verify the back-end used, you
can inspect the variable :code:`cudarray._backend`::

    import cudarray
    print(cudarray._backend)

You can force the back-end to CUDA by setting the environment variable
:code:`CUDARRAY_BACKEND` before importing CUDArray/DeepPy::

    import os
    os.environ['CUDARRAY_BACKEND'] = 'cuda'
    import deeppy

If the CUDA back-end fails to start, an exception will be raised with an error
message

.. _CUDArray: http://github.com/andersbll/cudarray


CUDA back-end installation problems
-----------------------------------
For some Python configurations, the shared library :code:`libcudarray.so`
cannot be located and an error will be raised::

    ImportError: libcudarray.so: cannot open shared object file: No such file or directory

In that case, try setting :code:`LD_LIBRARY_PATH` to include the directory
where :code:`libcudarray.so` is installed. See also this
`issue <http://github.com/andersbll/cudarray/issues/10>`_.
