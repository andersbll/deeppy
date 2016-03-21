import sys
import traceback
import cudarray as ca
from ...base import CollectionMixin
from . import digraph
from ..base import (
    Op, NoBPropMixin, NoFPropMixin, SplitMixin, Output
)


class ExprSplit(Op, SplitMixin):
    def __init__(self, n_splits):
        if n_splits <= 1:
            raise ValueError('n_splits should be >1')
        self.n_splits = n_splits

    def __call__(self, x):
        self.x = x
        self.inputs = [x]
        self.outputs = [Output()(self) for i in range(self.n_splits)]
        self.bpropable = x.bpropable
        return self.outputs

    def setup(self):
        self.shape = self.x.shape
        for i in range(self.n_splits):
            self.outputs[i].shape = self.shape
            self.outputs[i].array = self.x.array
            self.outputs[i].bpropable = self.bpropable
            if self.bpropable:
                self.outputs[i].grad_array = ca.zeros(self.shape)

    def fprop(self):
        for i in range(self.n_splits):
            self.outputs[i].array = self.x.array

    def bprop(self):
        ca.copyto(self.x.grad_array, self.outputs[0].grad_array)
        for i in range(1, self.n_splits):
            self.x.grad_array += self.outputs[i].grad_array


def _require_list(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, '__iter__'):
        return list(obj)
    else:
        return [obj]


def node_exception_msg(node):
    msg = 'Exception occurs in node %s' % node.__class__.__name__
    if node.inputs:
        msg += ' with inputs:'
        for n in node.inputs:
            shape = n.shape
            if isinstance(n, Output):
                n = n.inputs[0]
                if isinstance(n, ExprSplit):
                    n = n.inputs[0]
            name = n.__class__.__name__
            msg += '\n    %s, shape: %s' % (name, shape)
    return msg


def traceback_str():
    exc_info = sys.exc_info()
    trace = traceback.format_exception(*exc_info)
    return ''.join(trace)


def build_graph(sinks):
    graph = digraph.DiGraph()
    nodes = set(sinks)
    seen = set(nodes)
    while nodes:
        node = nodes.pop()
        for neighbor in node.inputs:
            graph.add_edge(neighbor, node)
            if neighbor not in seen:
                nodes.add(neighbor)
                seen.add(neighbor)
    return graph


class ExprGraph(CollectionMixin):
    def __init__(self, sinks):
        self.sinks = _require_list(sinks)
        self._initialized = False
        self._fprop_top = None
        self._bprop_top = None
        self.graph = None

    def _setup_nodes(self, nodes):
        for node in nodes:
            try:
                node.setup()
            except:
                raise Exception('\n' + traceback_str() + '\n\n' +
                                node_exception_msg(node))

    def setup(self):
        graph = build_graph(self.sinks)

        # Insert ExprSplit nodes
        for node, out_degree in graph.out_degree():
            if out_degree <= 1 or out_degree - len(node.inputs) == 0 or \
               not node.bpropable or isinstance(node, SplitMixin):
                continue
            split = ExprSplit(out_degree)
            split_exprs = split(node)
            for i, (_, in_node) in enumerate(list(graph.edges([node]))):
                graph.remove_edge(node, in_node)
                new_inputs = [split_exprs[i] if n is node else n
                              for n in in_node.inputs]
                in_node(*new_inputs)
                graph.add_edge(split, split_exprs[i])
                graph.add_edge(split_exprs[i], in_node)
            graph.add_edge(node, split)

        # Prepare fprop and bprop orderings
        fprop_top = digraph.topsort(graph)
        self._setup_nodes(fprop_top)

        # We need to rebuild graph because setup() may change the graph to
        # facilitate broadcasting operations
        # TODO: figure out if this should be disallowed
        graph = build_graph(self.sinks)
        fprop_top = digraph.topsort(graph)
        fprop_top = [n for n in fprop_top if not isinstance(n, NoFPropMixin)]

        graph_rev = digraph.reverse(graph)
        bprop_top = digraph.topsort(graph_rev)
        bprop_top = [n for n in bprop_top
                     if n.bpropable and not isinstance(n, NoBPropMixin)]

        self.graph = graph
        self._fprop_top = fprop_top
        self._bprop_top = bprop_top
        self._initialized = True

    @property
    def collection(self):
        return self.graph.nodes()

    def fprop(self):
        for node in self._fprop_top:
            node.fprop()

    def bprop(self):
        for node in self._bprop_top:
            node.bprop()
