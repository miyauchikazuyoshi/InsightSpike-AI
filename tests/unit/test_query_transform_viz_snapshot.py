from insightspike.visualization.query_transform_viz import snapshot


def test_query_transform_snapshot_summary_keys():
    hist = [
        {"confidence": 0.2, "transformation_magnitude": 0.1, "insights": []},
        {"confidence": 0.4, "transformation_magnitude": 0.3, "insights": ["a"]},
        {"confidence": 0.6, "transformation_magnitude": 0.5, "insights": ["b", "c"]},
    ]
    snap = snapshot(hist)
    assert set(["steps", "last_confidence", "last_magnitude", "total_insights"]).issubset(
        snap.keys()
    )
    assert snap["steps"] == 3
    assert snap["last_confidence"] == 0.6
    assert snap["last_magnitude"] == 0.5
    assert snap["total_insights"] == 3

