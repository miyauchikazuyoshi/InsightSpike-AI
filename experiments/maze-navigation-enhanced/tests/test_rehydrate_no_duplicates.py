import os, sys, tempfile, shutil, numpy as np
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from navigation.maze_navigator import MazeNavigator  # type: ignore

def _open_maze(w=6,h=6):
    return np.zeros((h,w),dtype=int)

def test_rehydrate_no_duplicates_second_call():
    tmpdir = tempfile.mkdtemp(prefix='maze_rehydrate_dup_')
    try:
        maze=_open_maze()
        start=(0,0); goal=(5,5)
        nav=MazeNavigator(maze,start,goal,wiring_strategy='simple',simple_mode=True,
                          enable_flush=True,flush_interval=1,max_in_memory=1000,max_in_memory_positions=1,
                          persistence_dir=tmpdir)
        nav.run(max_steps=35)
        # Choose a position known to have evictions (any from catalog)
        assert len(nav._evicted_catalog) > 0  # type: ignore[attr-defined]
        pos = next(iter(nav._evicted_catalog.values()))['position']  # type: ignore[attr-defined]
        pos = tuple(pos)
        rebuilt1 = nav.rehydrate_position(pos)  # first explicit attempt
        rebuilt2 = nav.rehydrate_position(pos)  # immediate second attempt should rebuild 0
        assert rebuilt2 == 0, f"Expected 0 rebuilt second call, got {rebuilt2}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
