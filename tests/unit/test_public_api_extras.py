def test_public_api_config_wrappers_import():
    from insightspike.public import load_config, get_config_summary
    assert callable(load_config)
    assert callable(get_config_summary)


def test_public_api_config_summary_noarg():
    from insightspike.public import get_config_summary
    summary = get_config_summary()
    assert isinstance(summary, dict)
    # memory key may or may not be present depending on loader behavior; just ensure dict


def test_public_api_create_datastore_memory():
    from insightspike.public import create_datastore
    ds = create_datastore("memory")
    # Minimal surface checks
    assert hasattr(ds, "save_episodes")
    assert hasattr(ds, "load_episodes")

