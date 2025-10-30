from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
from insightspike.maze_experimental.navigators.gediq_navigator import GeDIGNavigator

class DummyObs:
    def __init__(self,pos):
        self.position=pos
        self.possible_moves=[0]
        self.is_goal=False; self.is_junction=True; self.is_dead_end=False; self.hit_wall=False
        self.num_paths=3
    def to_features(self):
        return {'type':'junction','position':self.position}

class DummyMaze:
    ACTIONS={0:(1,0),1:(-1,0),2:(0,1),3:(0,-1)}

def test_wiring_candidate_selection():
    cfg = MazeNavigatorConfig(use_refactored_gedig=True)
    nav = GeDIGNavigator(cfg)
    # seed some memory nodes by simulating visits
    for p in [(0,0),(0,1),(1,0)]:
        nav.memory_nodes[p] = nav.memory_nodes.get(p) or type('N',(),{'position':p,'features':{'type':'junction'},'vector':None,'creation_energy':1.0,'visits':1,'last_visited':0})()
    # candidates creating potential structural benefit
    candidates = [((0,0),(1,0)), ((0,1),(1,0)), ((0,0),(0,2))]
    chosen = nav.evaluate_wiring_candidates(candidates)
    assert chosen in candidates
