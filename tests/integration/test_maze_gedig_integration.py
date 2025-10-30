import networkx as nx

def test_simple_structural_improvement_energy_reduction():
    """Minimal integration: structural improvement should reduce energy when adding new node."""
    from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
    from insightspike.maze_experimental.navigators.gediq_navigator import GeDIGNavigator

    # Dummy observation / maze mocks
    class DummyMaze:
        ACTIONS = {0:(1,0),1:(-1,0),2:(0,1),3:(0,-1)}
    class Obs:
        def __init__(self,pos):
            self.position=pos
            self.possible_moves=[0]
            self.is_goal=False; self.is_junction=True; self.is_dead_end=False; self.hit_wall=False
            self.num_paths=3
        def to_features(self):
            return {'type':'junction','position':self.position}
    maze = DummyMaze()
    cfg = MazeNavigatorConfig(use_refactored_gedig=True, enable_dual_evaluate=False)
    nav = GeDIGNavigator(cfg)
    obs = Obs((0,0))
    # First decide action builds initial memory
    nav.decide_action(obs, maze)
    # Create second observation to induce structural improvement
    obs2 = Obs((1,0))
    energy_before = nav._evaluate_action_energy((0,0),0,[],maze,obs)
    energy_after = nav._evaluate_action_energy((0,0),0,[],maze,obs2)
    # After exploring new node structural improvement should (likely) reduce energy
    assert energy_after <= energy_before + 1e-6
