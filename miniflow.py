import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node self will receive values
        self.inbound_nodes = inbound_nodes
        # Nodes from which this Node self will pass values
        self.outbound_nodes = []
        # For each inbound node, append this Node as outbound Node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        self.value = None

    def forward(self):
        """
            Forward Propagation.

            Compute the output value based on `inbound_nodes` and store
            the result in self.value
        """
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # Input Node has no inbound_nodes
        Node.__init__(self)

    def forward(self, value=None):
        """
            Input Node is the only Node where we can pass self.value to
            as argument to forward. Every other Node must rely on self.inbound_nodes
        """
        if value is not None:
            self.value = value

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 0
        for n in range(len(self.inbound_nodes)):
            self.value += self.inbound_nodes[n].value

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for n in range(len(self.inbound_nodes)):
            self.value *= self.inbound_nodes[n].value

def topological_sort(feed_dict):
    """
        Sort generic nodes in topological order using Kahn's Algorithm
    """
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop()
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in'] == 0):
                S.add(m)
    return L

def forward_pass(output_node, sorted_nodes):
    """
        Performs a forward pass through a list of sorted nodes
    """
    for n in sorted_nodes:
        n.forward()
    return output_node.value
