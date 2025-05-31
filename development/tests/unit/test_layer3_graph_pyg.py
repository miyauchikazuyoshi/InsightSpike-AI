import numpy as np

def test_build_graph(tmp_path):
    from insightspike.layer3_graph_pyg import build_graph
    
    vecs = np.zeros((2,2))
    data, edge_index = build_graph(vecs, dest=tmp_path/'g.pt')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'x')
    # Verify the file was created
    assert (tmp_path/'g.pt').exists()
