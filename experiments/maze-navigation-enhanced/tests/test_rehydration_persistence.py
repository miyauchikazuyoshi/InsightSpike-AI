import os, sys, tempfile, shutil
import numpy as np

# Ensure src on path
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from navigation.maze_navigator import MazeNavigator  # type: ignore

def _open_maze(w=5, h=5):
    # 0 = open cell
    return np.zeros((h, w), dtype=int)

def test_rehydration_across_process_with_persistence():
    tmpdir = tempfile.mkdtemp(prefix='maze_persist_')
    try:
        maze = _open_maze(6,6)
        start=(0,0); goal=(5,5)
        # First navigator: force aggressive eviction (position cap=1, flush every step)
        nav = MazeNavigator(maze, start, goal,
                            wiring_strategy='simple', simple_mode=True,
                            enable_flush=True, flush_interval=1,
                            max_in_memory=1000, max_in_memory_positions=1,
                            persistence_dir=tmpdir)
        nav.run(max_steps=40)
        stats1 = nav.get_statistics()
        assert stats1['flush_enabled'] is True
        # Ensure at least some evictions happened and catalog persisted
        assert stats1['episodes_evicted_total'] > 0, 'No episodes evicted; test setup insufficient'
        catalog_path = os.path.join(tmpdir, 'evicted_catalog.jsonl')
        assert os.path.isfile(catalog_path), 'Catalog file not created'
        # Pick one evicted position from in-memory catalog
        assert len(nav._evicted_catalog) > 0  # type: ignore[attr-defined]
        first_meta = next(iter(nav._evicted_catalog.values()))  # type: ignore[attr-defined]
        target_pos = tuple(first_meta['position'])  # type: ignore[index]
        # Start fresh navigator (simulating restart) pointing to same persistence_dir
        nav2 = MazeNavigator(maze, start, goal,
                             wiring_strategy='simple', simple_mode=True,
                             enable_flush=True, flush_interval=5,
                             max_in_memory=1000, max_in_memory_positions=2,
                             persistence_dir=tmpdir)
        # Catalog should be loaded
        assert len(nav2._evicted_catalog) > 0  # type: ignore[attr-defined]
        # Force rehydration without moving (if target is start, step so we revisit later)
        rebuilt = nav2.rehydrate_position(target_pos)  # type: ignore[attr-defined]
        assert rebuilt >= 0  # may be 0 if target position already has all directions
        # Move to target pos to trigger lazy rehydrate (will attempt on step())
        # Simple walk: brute force step until path includes target or max tries
        for _ in range(50):
            if nav2.current_pos == target_pos:
                break
            nav2.step()
        # After visiting, attempt explicit rehydrate again; should not create duplicates
        before_ids = {e.episode_id for e in nav2.episode_manager.episodes.values()}
        rebuilt2 = nav2.rehydrate_position(target_pos)  # type: ignore[attr-defined]
        after_ids = {e.episode_id for e in nav2.episode_manager.episodes.values()}
        assert rebuilt2 == 0, 'Second rehydrate produced duplicates'
        assert before_ids == after_ids, 'Episode IDs changed unexpectedly after second rehydrate'
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
