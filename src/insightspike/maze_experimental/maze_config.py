"""Configuration for maze experiments."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class MazeConfig(BaseModel):
    """Configuration for maze environment."""
    # Maze settings
    size: tuple[int, int] = Field(default=(20, 20), description="Maze size (height, width)")
    maze_type: Literal["simple", "complex", "random"] = Field(default="simple", description="Type of maze to generate")
    
    # Episode settings
    max_steps: int = Field(default=1000, description="Maximum steps per episode")
    num_episodes: int = Field(default=100, description="Number of episodes to run")
    
    # Rendering
    render_mode: Optional[Literal["ascii", "matplotlib", "none"]] = Field(default="ascii", description="How to render the maze")
    render_frequency: int = Field(default=10, description="Render every N episodes")
    save_animations: bool = Field(default=True, description="Save animations of trajectories")


class MazeNavigatorConfig(BaseModel):
    """Configuration for maze navigation with geDIG."""
    # Memory settings
    node_creation_cost: float = Field(default=0.0, description="Energy cost of creating new node (PoC default=0 for pure geDIG-driven behavior)")
    
    # Search settings
    search_radius: float = Field(default=5.0, description="Radius for sphere search")
    donut_inner_radius: float = Field(default=0.0, description="Inner radius for donut search")
    donut_outer_radius: float = Field(default=3.0, description="Outer radius for donut search")
    
    # Navigation strategy
    exploration_epsilon: float = Field(default=0.1, description="Epsilon for epsilon-greedy exploration")
    wall_penalty: float = Field(default=0.0, description="Penalty for walls ahead (shaping disabled by default)")
    unknown_bonus: float = Field(default=0.0, description="Bonus for exploring unknown areas (shaping disabled by default)")
    
    # geDIG parameters
    w_ged: float = Field(default=1.0, description="Weight for Graph Edit Distance")
    k_ig: float = Field(default=2.0, description="Weight for Information Gain")
    temperature: float = Field(default=1.0, description="Temperature for action selection")
    ged_weight: float = Field(default=1.0, description="Alias for w_ged")
    ig_weight: float = Field(default=2.0, description="Alias for k_ig")
    
    # Wake-Sleep settings
    sleep_interval: int = Field(default=10, description="Episodes between sleep phases")
    sleep_optimization_steps: int = Field(default=50, description="Optimization steps during sleep")
    
    # Feature embedding
    feature_dim: int = Field(default=16, description="Dimension of feature embeddings")
    use_pretrained_embedder: bool = Field(default=False, description="Use pretrained sentence embedder")

    # --- Day2 geDIG refactor integration flags ---
    use_refactored_gedig: bool = Field(default=True, description="Use refactored GeDIGCore via feature flag")
    enable_dual_evaluate: bool = Field(default=False, description="Run legacy + ref calculator and track divergence")
    dual_delta_threshold: float = Field(default=0.3, description="Warn if dual evaluate delta exceeds this value")
    structural_improvement_weight: float = Field(default=0.5, description="Weight to reduce action energy by structural improvement (positive improvement lowers energy)")
    # Spike outcome derivation mode (Day3)
    spike_outcome_mode: str = Field(
        default="mirror",
        description="Derive ground-truth spike outcome: mirror (prediction), structural_positive (structural_improvement>0), spike_threshold (gedig_value<threshold)."
    )
    # --- Day4: expose core spike detection parameters to allow test-time tuning ---
    tau_s: float = Field(default=0.15, description="Spike structural_improvement threshold (forwarded to GeDIGCore)")
    tau_i: float = Field(default=0.25, description="Spike ig_z_score threshold (forwarded to GeDIGCore)")
    spike_detection_mode: str = Field(default="and", description="Spike detection mode: and|or|threshold (forwarded to GeDIGCore)")


class MazeExperimentConfig(BaseModel):
    """Full configuration for maze experiments."""
    # Sub-configurations
    maze: MazeConfig = Field(default_factory=MazeConfig)
    navigator: MazeNavigatorConfig = Field(default_factory=MazeNavigatorConfig)
    
    # Experiment settings
    experiment_name: str = Field(default="maze_gediq", description="Name of the experiment")
    algorithms: list[str] = Field(
        default=["random", "dfs", "astar", "gediq"],
        description="Algorithms to compare"
    )
    
    # Logging
    log_metrics: bool = Field(default=True, description="Log metrics during training")
    save_checkpoints: bool = Field(default=True, description="Save model checkpoints")
    checkpoint_interval: int = Field(default=20, description="Save checkpoint every N episodes")
    
    # Random seed
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")