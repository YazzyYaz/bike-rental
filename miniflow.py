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

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        self.x_value = self.inbound_nodes[0].value
        self.y_value = self.inbound_nodes[1].value
        self.value = self.x_value + self.y_value
