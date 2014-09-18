import os


backend = os.getenv('DEEPPY_BACKEND', 'numpy').lower()

if backend == 'numpy':
    from .numpy import *
elif backend == 'cudarray':
    try:
        from cudarray import *
    except:
        raise ImportError('Failed to load CUDA back-end.')
else:
    raise ValueError('Invalid back-end "%s" specified.' % backend)
