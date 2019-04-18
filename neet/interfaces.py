"""
.. currentmodule:: neet.interfaces

.. testsetup:: interfaces

    from neet.automata import ECA
    from neet.interfaces import *
    from neet.statespace import StateSpace

API Documentation
-----------------
"""
from .statespace import StateSpace
from abc import ABCMeta, abstractmethod
import networkx as nx
import six


@six.add_metaclass(ABCMeta)
class Network(object):
    def __init__(self, size, metadata=None):
        if not isinstance(size, int):
            raise TypeError("Network size is not an int")
        elif size < 1:
            raise ValueError("Network size is negative")

        if metadata is None:
            metadata = dict()
        elif not isinstance(metadata, dict):
            raise TypeError('metadata is not a dict')

        self._size = size
        self._metadata = metadata

    @property
    def size(self):
        return self._size

    @property
    def metadata(self):
        return self._metadata

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def neighbors_in(self, index, *args, **kwargs):
        pass

    @abstractmethod
    def neighbors_out(self, index, *args, **kwargs):
        pass

    def neighbors(self, index, direction='both', *args, **kwargs):
        if direction not in ('in', 'out', 'both'):
            raise ValueError('direction must be "in", "out" or "both"')

        if direction == 'in':
            return self.neighbors_in(index, *args, **kwargs)
        elif direction == 'out':
            return self.neighbors_out(index, *args, **kwargs)
        else:
            inputs = self.neighbors_in(index, *args, **kwargs)
            outputs = self.neighbors_out(index, *args, **kwargs)
            return inputs.union(outputs)

    @abstractmethod
    def to_networkx_graph(self, *args, **kwargs):
        edges = [(i, j) for i in range(self.size) for j in self.neighbors_out(i)]

        return nx.DiGraph(edges, **kwargs)

    def draw(self, filename=None, *args, **kwargs):
        graph = self.to_networkx_graph(*args, **kwargs)
        nx.nx_agraph.view_pygraphviz(graph, prog='circo', path=filename)


class BooleanNetwork(Network):
    def __init__(self, size, metadata=None):
        super(BooleanNetwork, self).__init__(size, metadata)
        self._state_space = StateSpace(self.size, base=2)

    def state_space(self):
        return self._state_space


Network.register(BooleanNetwork)
