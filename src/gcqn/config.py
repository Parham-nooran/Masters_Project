import argparse
from dataclasses import dataclass


@dataclass
class GCQNConfig:
    """Configuration for Q-value Guided Growth with Lazy Pruning."""

    env_type: str = "dmcontrol"
    task: str = "walker_walk"
    seed: int = 42
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

    initial_bins: int = 5
    final_bins: int = 9
    confidence_threshold: int = 50
    temperature_decay: float = 0.995

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
    metrics_save_interval: int = 100
    log_interval: int = 5
    detailed_log_interval: int = 25

    action_penalty_coeff: float = 0.0

    load_checkpoints: str = None
    load_metrics: str = None


def create_config_from_args(args):
    """Create config from command line arguments."""
    config = GCQNConfig()

    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    return config


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train GCQN Agent"
    )

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
    parser.add_argument("--seed", type=int, default=42)
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
    """Add action space adaptation arguments."""
    parser.add_argument("--initial-bins", type=int, default=5)
    parser.add_argument("--final-bins", type=int, default=9)
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=50,
        help="Episodes before enabling weighted selection"
    )
    parser.add_argument(
        "--temperature-decay",
        type=float,
        default=0.995,
        help="Temperature decay rate for softmax selection"
    )


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
    parser.add_argument("--detailed-log-interval", type=int, default=25)
    parser.add_argument("--metrics-save-interval", type=int, default=100)


def _add_checkpoint_arguments(parser):
    """Add checkpoint-related arguments."""
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--load-checkpoints", type=str, default=None)
    parser.add_argument("--load-metrics", type=str, default=None)