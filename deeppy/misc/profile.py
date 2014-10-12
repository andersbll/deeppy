import time
import cudarray as ca


def profile(neuralnet, X, Y, reps=100):
    neuralnet._setup(X, Y)
    layer_input = ca.array(X)
    for layer_idx, layer in enumerate(neuralnet.layers[:-1]):
        start_time = time.time()
        for _ in range(reps):
            layer.fprop(layer_input, 'train')
        end_time = time.time()
        fprop_duration = float(end_time - start_time)/reps

        layer_output = layer.fprop(layer_input, 'train')
        start_time = time.time()
        for _ in range(reps):
            layer.bprop(layer_output)
        end_time = time.time()
        bprop_duration = float(end_time - start_time)/reps

        print('%s:   \tfprop: %.6f s \t bprop: %.6f s'
              % (layer.name, fprop_duration, bprop_duration))
        layer_input = layer_output
