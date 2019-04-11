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
        self.order = len(self.vertices)
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

    def single_source_shortest_path_length(self, source, cutoff=None):
        """Compute the shortest path lengths from source to all reachable nodes.
           Adapted from NetworkX

        Parameters
        ----------
        source : node
           Starting node for path

        Returns
        -------
        lengths : dict
            Dict keyed by node to shortest path length to source.
        """
        nextlevel = {source: 1}  # dict of nodes to check at next level
        seen = {}                  # level (number of hops) when seen in BFS
        level = 0                  # the current level
        lengths = {}

        while nextlevel:
            thislevel = nextlevel  # advance to next level
            nextlevel = {}         # and start a new list (fringe)
            for v in thislevel:
                if v not in seen:
                    seen[v] = level  # set the level of vertex v
                    # add neighbors of v
                    nextlevel.update({x: {} for x in self.as_dict[v]})
                    lengths[v] = level
            level += 1
        del seen
        return lengths

    def diameter(self):
        """Calculate the diameter of the graph
           Adapted from NetworkX

        The diameter of a graph is the greatest distance between any pair of vertices
        """
        order = self.order

        e = {}
        for n in range(order):
            length = self.single_source_shortest_path_length(n)
            if len(length) != order:
                print("Graph not connected: infinite path length")

            e[n] = max(length.values())

        return max(e.values())

    def dim(self):
        """Calculate the finite-dimension

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
        n = self.order
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
                # dim, dia = # pre-calculated values
            #plt.plot(dim, dia, 'k.')
        plt.plot(self.dim(), self.diameter(), 'C0o')
