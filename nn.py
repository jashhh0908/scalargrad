import random
from typing import Any
from engine import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.nonlin = nonlin
        self.w = []
        for i in range(nin):
            num = random.uniform(-1.0, 1.0)
            weight = Value(num)
            self.w.append(weight)
        rbias = random.uniform(-1.0, 1.0)
        self.b = Value(rbias)

    def __call__(self, inputs):
        activation = self.b
        for i in range(len(inputs)):
            w = self.w[i]
            x = inputs[i]
            activation += (w*x)
        # print(f"Raw sum before ReLU: {activation.data}") 
        return activation.relu() if self.nonlin else activation
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, n_neurons, nonlin=True):
        self.neurons = []
        for i in range(n_neurons):
            new_neuron = Neuron(nin, nonlin=nonlin)
            self.neurons.append(new_neuron)

    def __call__(self, inputs):
        outputs = []
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron_output = neuron(inputs)
            outputs.append(neuron_output)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
        
    def parameters(self):
        layer_param = []
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            layer_param.extend(neuron.parameters())
        return layer_param

class MLP:
    def __init__(self, nin, nouts) :
        size = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            if i == len(nouts) - 1:
                layer = Layer(size[i], size[i+1], nonlin=False)
            else: 
                layer = Layer(size[i], size[i+1], nonlin=True)
            self.layers.append(layer)

    def __call__(self, inputs):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = layer(inputs) # output of current layer becomes input for second layer
        return inputs
    
    def parameters(self):
        mlp_param = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            mlp_param.extend(layer.parameters())
        return mlp_param

if __name__ == "__main__":
    n = MLP(3, [4,4,1])
    inputs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    outputs = [1.0, -1.0, -1.0, 1.0]
    for step in range(20):
        predictions = []
        for x in inputs:
            predictions.append(n(x))

        loss = Value(0.0)
        for i in range(len(outputs)):
            ans = outputs[i]
            prediction = predictions[i]
            error = ans - prediction
            loss += (error ** 2)

        for p in n.parameters():
            p.grad = 0.0

        loss.backward()

        for p in n.parameters():
            p.data = p.data - (0.025 * p.grad)

        print(f"Step {step}: Loss = {loss.data}")