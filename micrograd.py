import math
import random

class Valoare:
  
    def __init__(self, data, _copii=(), _op=''):
        # Constructorul clasei Valoare.
        # data: Valoarea reală a obiectului (valoarea propriu-zisă).
        # _copii: O listă de noduri copii ale obiectului în cadrul graficului de calcul.
        # _op: Eticheta operației efectuate asupra acestui nod.
        # eticheta: O etichetă opțională pentru a identifica variabila (folosită în scopuri de debug sau documentare).
        self.data = data
        self.grad = 0.0  # Gradientul inițializat la 0.0
        self._inapoi = lambda: None  # Funcție de backward inițializată la o lambda care nu face nimic.
        self._precedent = set(_copii)  # Setul de noduri copii ale acestui nod.
        self._op = _op  # Eticheta operației efectuate.

    def __repr__(self):
        # Reprezentarea string a obiectului Valoare.
        return f"Valoare(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Valoare) else Valoare(other)
        out = Valoare(self.data + other.data, (self, other), '+')
        
        # Backward pentru adunare.
        def _inapoi():
            # Calculul gradientului pentru adunare.
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._inapoi = _inapoi
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Valoare) else Valoare(other)
        out = Valoare(self.data * other.data, (self, other), '*')
        
        # Backward pentru înmulțire.
        def _inapoi():
            # Calculul gradientului pentru înmulțire.
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._inapoi = _inapoi
          
        return out
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)

    def __radd__(self, other): # other + self
        return self + other
        
    
    def __pow__(self, other):
        out = Valoare(self.data**other, (self,), f'**{other}')

        # Backward pentru ridicare la putere.
        def _inapoi():
            # Calculul gradientului pentru ridicare la putere.
            # d(out)/d(self) = other * self.data^(other-1), d(out)/d(other) = self.data^other * ln(self.data)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._inapoi = _inapoi

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Valoare(t, (self, ), 'tanh')
        
        # Backward pentru tangenta hiperbolică.
        def _inapoi():
            # Calculul gradientului pentru tangenta hiperbolică.
            # d(out)/d(self) = 1 - tanh^2(x)
            self.grad += (1 - t**2) * out.grad
        out._inapoi = _inapoi
        
        return out
    
    def exponențiala(self):
        x = self.data
        out = Valoare(math.exp(x), (self, ), 'exp')
        
        # Backward pentru funcția exponențială.
        def _inapoi():
            # Calculul gradientului pentru funcția exponențială.
            # d(out)/d(self) = exp(x)
            self.grad += out.data * out.grad 
        out._inapoi = _inapoi
        
        return out
    
    def relu(self):
        out = Valoare(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _inapoi():
            self.grad += (out.data > 0) * out.grad
        out._backward = _inapoi

        return out
     
    def inapoi(self):
        # Inițierea pasului de backward. Construirea unei liste topologice și apoi propagarea gradientelor înapoi.

        topo = []  # Lista topologică pentru a păstra ordinea corectă a nodurilor în backward.
        vizitate = set()  # Set pentru a ține evidența nodurilor vizitate.
        
        # Funcție auxiliară pentru construirea listei topologice.
        def construieste_topo(v):
            if v not in vizitate:
                vizitate.add(v)
                for copil in v._precedent:
                    construieste_topo(copil)
                topo.append(v)
        
        # Apelarea funcției pentru a construi lista topologică pornind de la nodul curent.
        construieste_topo(self)
        
        # Inițierea gradientului la 1.0 pentru nodul curent.
        self.grad = 1.0
        
        # Propagarea gradientelor înapoi, în ordinea inversă a listei topologice.
        for nod in reversed(topo):
            nod._inapoi()


class Neuron:
    def __init__(self, nin):
       self.w = [Valoare(random.uniform(-1,1)) for _ in range(nin)]
       self.b = Valoare(random.uniform(-1,1))
    def __call__(self, x):
       act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
       out = act.tanh()
       return out
    # This method returns all the weights and the bias of the neuron.
    def parameters(self): 
       return self.w + [self.b]
# Class: Layer
# Initialization (__init__ method):

# nin is the number of inputs to each neuron in this layer.
# nout is the number of neurons in this layer.
# self.neurons creates a list of nout neurons, each having nin inputs.
# Forward Pass (__call__ method):

# When you give inputs to the layer, it feeds these inputs to each neuron in the layer.
# It collects the output from each neuron. If there's only one neuron, it returns the single output; otherwise, it returns a list of outputs.
# Parameters (parameters method):

# This method returns all the parameters (weights and biases) of all neurons in the layer.
class Layer:
   def __init__(self, nin, nout):
       self.neurons = [Neuron(nin) for _ in range(nout)]
   def __call__(self, x):
       outs = [n(x) for n in self.neurons]
       return outs[0] if len(outs) == 1 else outs
   def parameters(self):
       return [p for neuron in self.neurons for p in neuron.parameters()]
# Class: MLP (Multi-Layer Perceptron)
# Initialization (__init__ method):

# nin is the number of inputs to the first layer.
# nouts is a list where each element represents the number of neurons in the respective layer.
# self.layers creates a list of layers. The size of each layer is defined by sz, which combines nin and nouts.
# Forward Pass (__call__ method):

# When you give inputs to the MLP, it passes these inputs through each layer sequentially, with the output of one layer becoming the input to the next.
# It returns the final output after passing through all layers.
# Parameters (parameters method):

# This method returns all the parameters (weights and biases) of all neurons in all layers.
class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, 4.4]
n = MLP(3, [4, 4, 1])
print(*n.parameters(), sep='\n')

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
ypred = [n(x) for x in xs]
print(ypred)
loss = sum([(Valoare(ygt)-yout)**2 for ygt, yout in zip(ys, ypred)])
print(loss)
loss.inapoi()

for k in range(20):
  
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.inapoi()
  
  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
  
  print(k, loss.data)