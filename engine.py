class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.grad = 0.0
    
    def __repr__(self):
        return f"Value (data = {self.data})"

    def __add__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            other = Value(other)
        sum = self.data + other.data
        children = (self, other)
        output = Value(sum, children, '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output        
        
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            other = Value(other)
        difference = self.data - other.data
        children = (self, other)
        output = Value(difference, children, '-')

        def _backward():
            self.grad += output.grad
            other.grad += -output.grad

        output._backward = _backward
        return output        

    def __rsub__(self, other):
        return Value(other) - self

    def __mul__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            other = Value(other)
        product = self.data * other.data
        children = (self, other)
        output = Value(product, children, '*')
        
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __rmul__(self, other):
        return self * other

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float input supported"
        out_data = self.data ** other
        output = Value(out_data, (self, ), "**")
        def _backward():
            local_derivative = other * (self.data ** (other - 1))
            self.grad += local_derivative * output.grad
        
        output._backward = _backward
        return output
        
    def relu(self):
        out_data = self.data if self.data >= 0 else 0.0
        output = Value(out_data, (self, ), 'ReLU')
        
        def _backward():
            local_derivative = 1.0 if self.data > 0 else 0.0
            self.grad += local_derivative * output.grad

        output._backward = _backward
        return output