import os
import tempfile
import subprocess
import numpy as np
from ..base import Output
from . import digraph
from .exprgraph import (
    ExprSplit, ExprGraph, build_graph, _require_list, node_exception_msg,
    traceback_str
)


def draw(sinks, filepath, omit_splits=True, emph_nodes=[]):
    # Requires: networkx, graphviz, pygraphviz
    import networkx as nx

    sinks = _require_list(sinks)
    graph = build_graph(sinks)

    nx_graph = nx.DiGraph()
    node_labels = {}
    graph = digraph.copy(graph)
    if omit_splits:
        for node in list(graph.nodes()):
            if isinstance(node, (ExprSplit, Output)):
                for _, neighbor in list(graph.in_edges([node])):
                    graph.remove_edge(neighbor, node)
                    for _, in_neighbor in list(graph.edges([node])):
                        graph.add_edge(neighbor, in_neighbor)
                for _, neighbor in list(graph.edges([node])):
                    graph.remove_edge(node, neighbor)
                graph.remove_node(node)
    for node in digraph.topsort(graph):
        label = node.__class__.__name__
        if label not in node_labels:
            node_labels[label] = 0
        else:
            node_labels[label] += 1
        label += ' #' + str(node_labels[label])
        if node.shape is not None:
            label += ' ' + str(node.shape)
        color = 'black' if node.bpropable else 'grey'
        nx_graph.add_node(node, label=label, color=color)

    for node in emph_nodes:
        nx_graph.add_node(node, color='red')

    nx_graph.add_edges_from(graph.edges())
    for node in nx_graph.nodes():
        if not node.bpropable:
            for _, neighbor in nx_graph.edges([node]):
                nx_graph.add_edge(node, neighbor, color='grey')

    _, tmpfilepath = tempfile.mkstemp(suffix='.dot')
    nx.drawing.nx_agraph.write_dot(nx_graph, tmpfilepath)
    subprocess.call(['dot', '-Tpdf', tmpfilepath, '-o', filepath])
    os.remove(tmpfilepath)


class DebugExprGraph(ExprGraph):
    def _setup_nodes(self, nodes):
        visited = []
        for node in nodes:
            try:
                node.setup()
                visited.append(node.__class__.__name__)
            except:
                draw(self.sinks, 'debug_setup_trace.pdf', omit_splits=True,
                     emph_nodes=[node])
                raise Exception('\n' + traceback_str() + '\n\n' +
                                node_exception_msg(node) +
                                '\n\nNodes visited:\n' + str(visited))

    def fprop(self):
        visited = []
        for node in self._fprop_top:
            try:
                node.fprop()
                visited.append(node.__class__.__name__)
            except:
                draw(self.sinks, 'debug_fprop_trace.pdf', omit_splits=True,
                     emph_nodes=[node])
                raise Exception('\n' + traceback_str() +
                                '\n' + node_exception_msg(node) +
                                '\n\nNodes visited:\n' + str(visited))

    def bprop(self):
        visited = []
        for node in self._bprop_top:
            try:
                node.bprop()
                visited.append(node.__class__.__name__)
            except:
                draw(self.sinks, 'debug_bprop_trace.pdf', omit_splits=True,
                     emph_nodes=[node])
                raise Exception('\n' + traceback_str() +
                                '\n' + node_exception_msg(node) +
                                '\n\nNodes visited:\n' + str(visited))


class NANGuardExprGraph(ExprGraph):
    def fprop(self):
        visited = []
        for node in self._fprop_top:
            node.fprop()
            if node.array is not None:
                arr = np.array(node.array)
                if np.any(np.isnan(arr) + np.isinf(arr)):
                    draw(self.sinks, 'debug_fprop_nan.pdf', omit_splits=True,
                         emph_nodes=[node])
                    raise Exception('\n' + traceback_str() +
                                    '\n' + node_exception_msg(node) +
                                    '\n\nNodes visited:\n' + str(visited))
            visited.append(node.__class__.__name__)

    def bprop(self):
        visited = []
        for node in self._bprop_top:
            if node.grad_array is not None:
                arr = np.array(node.grad_array)
                if np.any(np.isnan(arr) + np.isinf(arr)):
                    draw(self.sinks, 'debug_bprop_nan.pdf', omit_splits=True,
                         emph_nodes=[node])
                    raise Exception('\n' + traceback_str() +
                                    '\n' + node_exception_msg(node) +
                                    '\n\nNodes visited:\n' + str(visited))
            node.bprop()
            visited.append(node.__class__.__name__)
