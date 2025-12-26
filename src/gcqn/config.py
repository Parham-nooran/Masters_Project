import argparse
from dataclasses import dataclass


@dataclass
class GCQNConfig:
    """Configuration for Hybrid CQN-GQN with all hyperparameters."""

    env_type: str = "dmcontrol"
    task: str = "walker_walk"
    seed: int = 0
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000

    use_pixels: bool = False
    ogbench_dataset_dir: str = "~/.ogbench/data"
    discount: float = 0.99
    n_step: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-4
    target_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000

    initial_bins: int = 2
    final_bins: int = 9
    unmasking_strategy: str = "hybrid"
    growth_check_interval: int = 50
    min_episodes_before_growth: int = 100

    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

    huber_delta: float = 1.0
    gradient_clip: float = 40.0

    per_alpha: float = 0.6
    per_beta: float = 0.4

    layer_size: int = 512
    layer_size_bottleneck: int = 50
    num_layers: int = 2

    checkpoint_interval: int = 500
    metrics_save_interval: int = 500
    log_interval: int = 5
    detailed_log_interval: int = 50
    eval_episodes: int = 10

    action_penalty_coeff: float = 0.0

    load_checkpoints: str = None
    load_metrics: str = None


def create_hybrid_config_from_args(args):
    """Create config from command line arguments."""
    config = GCQNConfig()

    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    return config


def parse_hybrid_args():
    """Parse command line arguments for hybrid training."""
    parser = argparse.ArgumentParser(description="Train Hybrid CQN-GQN")

    _add_environment_arguments(parser)
    _add_training_arguments(parser)
    _add_action_space_arguments(parser)
    _add_exploration_arguments(parser)
    _add_optimization_arguments(parser)
    _add_replay_buffer_arguments(parser)
    _add_network_arguments(parser)
    _add_logging_arguments(parser)
    _add_checkpoint_arguments(parser)

    return parser.parse_args()


def _add_environment_arguments(parser):
    """Add environment-related arguments."""
    parser.add_argument(
        "--env-type",
        type=str,
        default="dmcontrol",
        choices=["dmcontrol", "metaworld", "ogbench"],
    )
    parser.add_argument("--task", type=str, default="walker_walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--max-steps-per-episode", type=int, default=1000)
    parser.add_argument("--use-pixels", action="store_true")
    parser.add_argument("--ogbench-dataset-dir", type=str, default="~/.ogbench/data")


def _add_training_arguments(parser):
    """Add training hyperparameter arguments."""
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-update-period", type=int, default=100)


def _add_action_space_arguments(parser):
    """Add action space growth arguments."""
    parser.add_argument("--initial-bins", type=int, default=2)
    parser.add_argument("--final-bins", type=int, default=9)
    parser.add_argument(
        "--unmasking-strategy",
        type=str,
        default="hybrid",
        choices=["variance", "advantage", "hybrid"],
    )
    parser.add_argument("--growth-check-interval", type=int, default=50)
    parser.add_argument("--min-episodes-before-growth", type=int, default=100)


def _add_exploration_arguments(parser):
    """Add exploration strategy arguments."""
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--min-epsilon", type=float, default=0.01)


def _add_optimization_arguments(parser):
    """Add optimization-related arguments."""
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=40.0)
    parser.add_argument("--action-penalty-coeff", type=float, default=0.0)


def _add_replay_buffer_arguments(parser):
    """Add replay buffer configuration arguments."""
    parser.add_argument("--min-replay-size", type=int, default=1000)
    parser.add_argument("--max-replay-size", type=int, default=1000000)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)


def _add_network_arguments(parser):
    """Add neural network architecture arguments."""
    parser.add_argument("--layer-size", type=int, default=512)
    parser.add_argument("--layer-size-bottleneck", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)


def _add_logging_arguments(parser):
    """Add logging and evaluation arguments."""
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--detailed-log-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--metrics-save-interval", type=int, default=100)


def _add_checkpoint_arguments(parser):
    """Add checkpoint-related arguments."""
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--load-checkpoints", type=str, default=None)
    parser.add_argument("--load-metrics", type=str, default=None)