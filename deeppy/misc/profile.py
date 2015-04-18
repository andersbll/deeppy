import time
import cudarray as ca

from ..input import Input


def avg_running_time(fun, reps):
    # Memory allocation forces GPU synchronization
    ca.empty(1)
    start_time = time.time()
    for _ in range(reps):
        fun()
    ca.empty(1)
    return float(time.time() - start_time) / reps


def profile(neuralnet, input, reps=50):
    input = Input.from_any(input)
    neuralnet._setup(input)
    batch = next(input.supervised_batches())
    layer_input = batch[0]
    total_duration = 0
    for layer_idx, layer in enumerate(neuralnet.layers[:-1]):
        def fprop():
            layer.fprop(layer_input, 'train')
        fprop_duration = avg_running_time(fprop, reps)
        layer_output = layer.fprop(layer_input, 'train')

        def bprop():
            layer.bprop(layer_output)
        bprop_duration = avg_running_time(bprop, reps)
        print('%s:   \tfprop(): %.6f s \t bprop(): %.6f s'
              % (layer.name, fprop_duration, bprop_duration))
        layer_input = layer_output
        total_duration += fprop_duration + bprop_duration
    print('total_duration: %.6f s' % total_duration)

    def nn_bprop():
        neuralnet._update(batch)
    nn_bprop_duration = avg_running_time(nn_bprop, reps)
    print('neuralnet._bprop(): %.6f s' % nn_bprop_duration)
