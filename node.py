class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node self will receive values
        self.inbound_nodes = inbound_nodes
        # Nodes from which this Node self will pass values
        self.outbound_nodes = []
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
