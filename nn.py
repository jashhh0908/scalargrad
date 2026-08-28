import random
from engine import Value

class Neuron:
    def __init__(self, nin):
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
            b = self.b
            activation += (w*x)
        print(f"Raw sum before ReLU: {activation.data}") 
        output = activation.relu()
        return output

if __name__ == "__main__":
    n = Neuron(3)
    inputs = [2.0, 3.0, -1.0]

    output = n(inputs)
    print(output)