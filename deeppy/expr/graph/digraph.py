"""
Directed graph.

Implementation heavily based on code from the NetworkX package.
"""


class DiGraph(object):
    def __init__(self):
        self._pred = {}
        self._succ = {}

    def nodes(self):
        return self._succ.keys()

    def edges(self, nodes=None):
        if nodes is None:
            nodes = self.nodes()
        for n in nodes:
            for neighbor in self._succ[n]:
                yield n, neighbor

    def in_edges(self, nodes=None):
        if nodes is None:
            nodes = self.nodes()
        for n in nodes:
            for neighbor in self._pred[n]:
                yield n, neighbor

    def add_node(self, n):
        if n not in self._succ:
            self._succ[n] = set()
            self._pred[n] = set()

    def add_nodes(self, nodes):
        for n in nodes:
            self.add_node(n)

    def remove_node(self, node):
        try:
            neighbors = self._succ[node]
        except KeyError:
            raise ValueError("Node is not in the graph: %s" % node)
        for u in neighbors:
            self._pred[u].remove(node)
        for u in self._pred[node]:
            self._succ[u].remove(node)
        del self._succ[node]
        del self._pred[node]

    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        self._succ[u].add(v)
        self._pred[v].add(u)

    def add_edges(self, edges):
        for u, v in edges:
            self.add_edge(u, v)

    def remove_edge(self, u, v):
        try:
            self._succ[u].remove(v)
            self._pred[v].remove(u)
        except KeyError:
            raise ValueError('Edge %s-%s is not in graph.' % (u, v))

    def in_degree(self, nodes=None):
        if nodes is None:
            nodes = self.nodes()
        for node in nodes:
            neighbors = self._pred[node]
            yield node, len(neighbors)

    def out_degree(self, nodes=None):
        if nodes is None:
            nodes = self.nodes()
        for node in nodes:
            neighbors = self._succ[node]
            yield node, len(neighbors)

    def __contains__(self, n):
        return n in self.nodes()

    def __len__(self):
        return len(self.nodes())


def topsort(graph, nodes=None):
    if nodes is None:
        nodes = graph.nodes()
    else:
        nodes = reversed(list(nodes))

    def dfs(graph, seen, explored, v):
        seen.add(v)
        for w in graph._succ[v]:
            if w not in seen:
                dfs(graph, seen, explored, w)
            elif w in seen and w not in explored:
                raise ValueError('Graph contains a cycle.')
        explored.insert(0, v)

    seen = set()
    explored = []
    for v in nodes:
        if v not in explored:
            dfs(graph, seen, explored, v)
    return explored


def copy(graph):
    graph_copy = DiGraph()
    graph_copy.add_nodes(graph.nodes())
    graph_copy.add_edges(graph.edges())
    return graph_copy


def reverse(graph):
    graph_rev = copy(graph)
    graph_rev._pred, graph_rev._succ = graph_rev._succ, graph_rev._pred
    return graph_rev
