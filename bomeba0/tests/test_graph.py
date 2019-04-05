from ..utils.graph import Graph


def test_dist():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    graph = Graph(data)
    assert graph.dim() == 1.5849625007211563
    assert graph.chromatic() == 3
    assert graph.complement().chromatic() == 3
    assert graph.diameter() == 2
    assert graph.is_connected() == True
