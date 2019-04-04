from math import log, inf, nan
from itertools import product


class Graph:
    """Graph object

    Parameters
    ----------

    data : list of tuples
        Pairs of directly conected nodes in the graph

    """

    def __init__(self, data):
        self.vertices = list({i for pair in data for i in pair})
        self.N = len(self.vertices)
        self.edges = data
        self.as_dict = self.get_dict()

    def get_dict(self):
        """ Dictionary representation of a graph.
        keys are nodes and values are all the nodes connected to the key-node
        """
        as_dict = {}
        for vertex in self.vertices:
            values = set()
            for pair in self.edges:
                if vertex in pair:
                    for e in pair:
                        if vertex != e:
                            values.add(e)
            as_dict[vertex] = values

        return as_dict

    def is_connected(self, vertices_encountered=None, start_vertex=None):
        """Returns True if the graph is connected.
        An undirected graph is connected when there is a path between every pair of vertices
        """
        if vertices_encountered is None:
            vertices_encountered = set()
        gdict = self.as_dict
        vertices = self.vertices
        if not start_vertex:
            start_vertex = vertices[0]
        vertices_encountered.add(start_vertex)
        if len(vertices_encountered) != len(vertices):
            for vertex in gdict[start_vertex]:
                if vertex not in vertices_encountered:
                    if self.is_connected(vertices_encountered, vertex):
                        return True
        else:
            return True
        return False

    def _find_all_paths(self, start_vertex, end_vertex, path=[]):
        """ Find all paths from `start_vertex` to `end_vertex` in graph"""
        graph = self.as_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self._find_all_paths(vertex,
                                                      end_vertex,
                                                      path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def diameter(self):
        """Calculates the diameter of the graph

        The diameter of a graph is the greatest distance between any pair of vertices
        """
        v = self.vertices
        all_pairs = [(v[i], v[j]) for i in range(len(v)-1)
                     for j in range(i+1, len(v))]
        shorter_paths = []
        for (first, second) in all_pairs:
            paths = self._find_all_paths(first, second)
            smallest = min([len(s) for s in paths])
            shorter_paths.append(smallest)

        diameter = max(shorter_paths) - 1
        return diameter

    def dim(self):
        """Calculates the finite-dimension

        For details see:
        https://arxiv.org/abs/1508.02946
        https://arxiv.org/abs/1607.08130
        """
        diameter = self.diameter()
        if diameter == 1:
            dim_f = inf
        elif diameter == 0:
            dim_f = 0
        elif not self.is_connected():
            dim_f = nan
        else:
            dim_f = log(self.complement().chromatic()) / log(diameter)

        return dim_f

    def complement(self):
        complement_data = set()
        v_set = set(self.vertices)
        for k, v in self.as_dict.items():
            c_set = v_set - v
            c_set.remove(k)
            for i in c_set:
                if k > i:
                    complement_data.add((k, i))
                else:
                    complement_data.add((i, k))

        return Graph(list(complement_data))

    def chromatic(self):
        n = self.N
        v = self.edges
        for i in range(1, n+1):
            for p in product(range(i), repeat=n):
                if(0 == len([x for x in v if(p[x[0]] == p[x[1]])])):
                    return i

    def plot_dim_f(self, context=None):
        import matplotlib.pyplot as plt
        if context is not None:
            if context == "glycan":  # assume we will have more contexts in the future
                pass
                # dim, dia =  # precalculated values
            #plt.plot(dim, dia, 'k.')
        plt.plot(self.dim(), self.diameter(), 'C0o')
