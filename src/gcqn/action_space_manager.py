import torch
import numpy as np
from collections import deque


class GCQNActionSpaceManager:
    """
    Q-value Guided Growth with Lazy Pruning action space manager.

    Algorithm phases:
    1. Uniform exploration: Build initial Q-estimates
    2. Weighted selection: Focus on high-Q bins
    3. Lazy pruning: Remove persistently low-value bins
    """

    def __init__(self, action_spec, initial_bins, final_bins, device,
                 confidence_threshold=50, temperature_decay=0.995):
        self.device = device
        self.action_min = torch.tensor(
            action_spec["low"], dtype=torch.float32, device=device
        )
        self.action_max = torch.tensor(
            action_spec["high"], dtype=torch.float32, device=device
        )
        self.action_dim = len(self.action_min)
        self.initial_bins = initial_bins
        self.final_bins = final_bins

        self.confidence_threshold = confidence_threshold
        self.temperature = 10.0
        self.temperature_decay = temperature_decay
        self.min_temperature = 0.5

        self.action_bins = self._create_full_action_grid()
        self.active_masks = self._initialize_uniform_coverage()

        self.metrics_tracker = QValueGuidedTracker(
            self.action_dim, self.final_bins, device
        )

        self.growth_history = []
        self.pruning_history = []
        self.current_phase = 1
        self.weighted_selection_enabled = False

    def _create_full_action_grid(self):
        """Create complete discretized grid across all dimensions."""
        bins_per_dim = []
        for dim in range(self.action_dim):
            dim_bins = torch.linspace(
                self.action_min[dim],
                self.action_max[dim],
                self.final_bins,
                device=self.device,
            )
            bins_per_dim.append(dim_bins)
        return torch.stack(bins_per_dim)

    def _initialize_uniform_coverage(self):
        """Initialize with uniform coverage across action space."""
        masks = torch.zeros(
            self.action_dim, self.final_bins, dtype=torch.bool, device=self.device
        )

        uniform_indices = self._compute_uniform_indices()
        for dim in range(self.action_dim):
            masks[dim, uniform_indices] = True

        return masks

    def _compute_uniform_indices(self):
        """Compute uniformly-spaced bin indices for initial coverage."""
        if self.initial_bins >= self.final_bins:
            return list(range(self.final_bins))

        step = (self.final_bins - 1) / (self.initial_bins - 1)
        return [int(round(i * step)) for i in range(self.initial_bins)]

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete action indices to continuous values."""
        discrete_actions = self._prepare_discrete_actions(discrete_actions)
        active_indices_per_dim = self._get_active_indices_per_dimension()
        actual_indices = self._map_to_actual_indices(
            discrete_actions, active_indices_per_dim
        )
        return self._gather_continuous_values(actual_indices)

    def _prepare_discrete_actions(self, discrete_actions):
        """Ensure discrete actions have proper shape and device."""
        if len(discrete_actions.shape) == 1:
            discrete_actions = discrete_actions.unsqueeze(0)
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)
        return discrete_actions

    def _get_active_indices_per_dimension(self):
        """Get list of active bin indices for each dimension."""
        return [
            torch.where(self.active_masks[dim])[0]
            for dim in range(self.action_dim)
        ]

    def _map_to_actual_indices(self, discrete_actions, active_indices_per_dim):
        """Map discrete indices to actual bin positions in full grid."""
        batch_size = discrete_actions.shape[0]
        actual_indices = torch.zeros_like(discrete_actions)

        for dim in range(self.action_dim):
            num_active = len(active_indices_per_dim[dim])
            clamped = torch.clamp(discrete_actions[:, dim], 0, num_active - 1)
            actual_indices[:, dim] = active_indices_per_dim[dim][clamped]

        return actual_indices

    def _gather_continuous_values(self, actual_indices):
        """Gather continuous action values from bins."""
        batch_size = actual_indices.shape[0]
        continuous_actions = torch.zeros(
            batch_size, self.action_dim, device=self.device
        )

        for dim in range(self.action_dim):
            bin_indices = actual_indices[:, dim].long()
            continuous_actions[:, dim] = self.action_bins[dim, bin_indices]

        return continuous_actions

    def get_active_q_values(self, q_values):
        """Extract Q-values for currently active bins."""
        batch_size = q_values.shape[0]
        active_q_list = []

        for dim in range(self.action_dim):
            active_indices = torch.where(self.active_masks[dim])[0]
            dim_active_q = q_values[:, dim, active_indices]
            active_q_list.append(dim_active_q)

        max_active = max(aq.shape[1] for aq in active_q_list)
        padded_active_q = self._pad_active_q_values(active_q_list, batch_size, max_active)

        return padded_active_q

    def _pad_active_q_values(self, active_q_list, batch_size, max_active):
        """Pad active Q-values to uniform shape."""
        padded = torch.full(
            (batch_size, self.action_dim, max_active),
            float("-inf"),
            device=self.device,
        )

        for dim, active_q in enumerate(active_q_list):
            num_active = active_q.shape[1]
            padded[:, dim, :num_active] = active_q

        return padded

    def update_metrics(self, q_values, actions):
        """Update tracking with new Q-values and actions."""
        self.metrics_tracker.update(q_values, actions, self.active_masks)

    def check_and_switch_to_weighted_selection(self, episode):
        """Check if should switch from uniform to weighted selection."""
        if self.weighted_selection_enabled:
            return False

        if not self.metrics_tracker.has_sufficient_data():
            return False

        if self.metrics_tracker.is_confident():
            self.weighted_selection_enabled = True
            self.current_phase = 2
            return True

        return False

    def get_weighted_action_probabilities(self, q_values):
        """
        Compute action selection probabilities weighted by Q-values.
        Uses temperature-based softmax for smooth exploration-exploitation.
        """
        if not self.weighted_selection_enabled:
            return None

        probabilities = []

        for dim in range(self.action_dim):
            active_indices = torch.where(self.active_masks[dim])[0]
            active_q = q_values[:, dim, active_indices]

            dim_probs = torch.softmax(active_q / self.temperature, dim=1)
            probabilities.append(dim_probs)

        return probabilities

    def decay_temperature(self):
        """Decay temperature for gradual shift from exploration to exploitation."""
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )

    def check_and_adapt(self, episode):
        """
        Check conditions and perform growth or pruning.

        Returns: (did_change, change_type)
        """
        if episode < self.confidence_threshold:
            return False, 'too_early'

        did_grow = False
        did_prune = False

        if episode % 50 == 0:
            bins_grown = self._perform_growth(episode)
            did_grow = bins_grown > 0

        if episode >= 150 and episode % 25 == 0:
            bins_pruned = self._perform_pruning(episode)
            did_prune = bins_pruned > 0

            if did_prune:
                self.current_phase = 3

        if did_grow:
            return True, 'growth'
        elif did_prune:
            return True, 'pruning'

        return False, 'none'

    def _perform_growth(self, episode):
        """Grow bins near high-Q regions."""
        if not self.metrics_tracker.has_sufficient_data():
            return 0

        mean_q_values = self.metrics_tracker.get_mean_q_values()
        bins_grown = 0

        for dim in range(self.action_dim):
            active_bins = torch.where(self.active_masks[dim])[0]

            if len(active_bins) >= self.final_bins:
                continue

            active_q = mean_q_values[dim, active_bins]
            top_k = max(1, len(active_bins) // 3)
            _, top_indices = torch.topk(active_q, k=top_k)
            top_bins = active_bins[top_indices]

            for bin_idx in top_bins:
                neighbors = self._get_growth_candidates(dim, bin_idx.item())

                for neighbor_idx in neighbors:
                    if not self.active_masks[dim, neighbor_idx]:
                        self.active_masks[dim, neighbor_idx] = True
                        bins_grown += 1

        if bins_grown > 0:
            self.growth_history.append({
                'episode': episode,
                'bins_grown': bins_grown
            })

        return bins_grown

    def _get_growth_candidates(self, dim, bin_idx):
        """Get neighboring bins that could be activated."""
        candidates = []

        if bin_idx > 0:
            candidates.append(bin_idx - 1)
        if bin_idx < self.final_bins - 1:
            candidates.append(bin_idx + 1)

        return candidates

    def _perform_pruning(self, episode):
        """Prune bins with persistently low Q-values and low visits."""
        if not self.metrics_tracker.has_sufficient_data():
            return 0

        mean_q_values = self.metrics_tracker.get_mean_q_values()
        recent_visits = self.metrics_tracker.get_recent_visit_counts()
        bins_pruned = 0

        for dim in range(self.action_dim):
            active_bins = torch.where(self.active_masks[dim])[0]

            if len(active_bins) <= 2:
                continue

            active_q = mean_q_values[dim, active_bins]
            active_visits = recent_visits[dim, active_bins]

            q_threshold = torch.quantile(active_q, 0.25)
            visit_threshold = 5

            for i, bin_idx in enumerate(active_bins):
                if (active_q[i] < q_threshold and
                        active_visits[i] < visit_threshold and
                        self._can_safely_prune(dim, bin_idx.item())):
                    self.active_masks[dim, bin_idx] = False
                    bins_pruned += 1

        if bins_pruned > 0:
            self.pruning_history.append({
                'episode': episode,
                'bins_pruned': bins_pruned
            })

        return bins_pruned

    def _can_safely_prune(self, dim, bin_idx):
        """Check if bin can be safely pruned without losing connectivity."""
        active_bins = torch.where(self.active_masks[dim])[0].cpu().numpy()

        if len(active_bins) <= 2:
            return False

        if bin_idx == active_bins[0] or bin_idx == active_bins[-1]:
            return len(active_bins) > 3

        left_neighbor = bin_idx - 1
        right_neighbor = bin_idx + 1

        has_active_neighbor = (
                (left_neighbor >= 0 and self.active_masks[dim, left_neighbor]) or
                (right_neighbor < self.final_bins and self.active_masks[dim, right_neighbor])
        )

        return has_active_neighbor

    def get_growth_info(self):
        """Get information about current growth state."""
        return {
            "current_phase": self.current_phase,
            "weighted_selection": self.weighted_selection_enabled,
            "temperature": self.temperature,
            "total_active_bins": self.active_masks.sum().item(),
            "total_possible_bins": self.action_dim * self.final_bins,
            "active_per_dimension": [
                self.active_masks[d].sum().item()
                for d in range(self.action_dim)
            ],
            "growth_events": len(self.growth_history),
            "pruning_events": len(self.pruning_history)
        }

    def get_visual_representation(self, logger=None):
        """Generate visual representation of active bins per dimension."""
        visualization = []

        mean_q = self.metrics_tracker.get_mean_q_values() if self.metrics_tracker.has_sufficient_data() else None

        for dim in range(self.action_dim):
            active_indices = torch.where(self.active_masks[dim])[0].cpu().numpy()
            dim_viz = self._create_dimension_visualization(dim, active_indices, mean_q)
            visualization.append(dim_viz)

            if logger:
                logger.info(f"  Dim {dim}: {dim_viz['ascii']}")
                logger.info(f"         Active: {dim_viz['active_bins']}/{self.final_bins} "
                            f"| Range: [{dim_viz['min_val']:.2f}, {dim_viz['max_val']:.2f}]")
                if mean_q is not None:
                    logger.info(f"         Avg Q: {dim_viz['avg_q']:.2f}")

        return visualization

    def _create_dimension_visualization(self, dim, active_indices, mean_q):
        """Create visualization for single dimension."""
        ascii_repr = []

        for bin_idx in range(self.final_bins):
            if bin_idx in active_indices:
                if mean_q is not None:
                    q_val = mean_q[dim, bin_idx].item()
                    active_q = mean_q[dim, active_indices]
                    q_mean = active_q.mean().item()

                    if q_val > q_mean + active_q.std().item():
                        ascii_repr.append("█")
                    elif q_val > q_mean:
                        ascii_repr.append("▓")
                    else:
                        ascii_repr.append("▒")
                else:
                    ascii_repr.append("█")
            else:
                ascii_repr.append("░")

        active_bins_values = self.action_bins[dim, active_indices].cpu().numpy()

        avg_q = 0.0
        if mean_q is not None:
            active_q = mean_q[dim, active_indices]
            avg_q = active_q.mean().item()

        return {
            "dimension": dim,
            "ascii": "".join(ascii_repr),
            "active_bins": len(active_indices),
            "min_val": active_bins_values.min() if len(active_bins_values) > 0 else 0,
            "max_val": active_bins_values.max() if len(active_bins_values) > 0 else 0,
            "active_indices": active_indices.tolist(),
            "avg_q": avg_q
        }

    def log_detailed_state(self, logger, episode):
        """Log comprehensive state information."""
        logger.info("=" * 80)
        logger.info(f"GCQN ACTION SPACE STATE at Episode {episode}")
        logger.info("=" * 80)

        phase_names = {
            1: "Uniform Exploration",
            2: "Weighted Selection",
            3: "Weighted + Pruning"
        }
        logger.info(f"Phase {self.current_phase}: {phase_names[self.current_phase]}")
        logger.info(f"Temperature: {self.temperature:.3f}")
        logger.info("")

        growth_info = self.get_growth_info()
        logger.info(f"Active bins: {growth_info['total_active_bins']}/{growth_info['total_possible_bins']} "
                    f"({100 * growth_info['total_active_bins'] / growth_info['total_possible_bins']:.1f}%)")
        logger.info(f"Growth events: {growth_info['growth_events']}")
        logger.info(f"Pruning events: {growth_info['pruning_events']}")
        logger.info("")

        logger.info("Active Bins per Dimension:")
        logger.info("Legend: █ = High Q  ▓ = Medium Q  ▒ = Low Q  ░ = Inactive")
        self.get_visual_representation(logger)
        logger.info("=" * 80)


class QValueGuidedTracker:
    """Tracks Q-values and visits for Q-value guided adaptation."""

    def __init__(self, action_dim, num_bins, device, history_size=100):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device
        self.history_size = history_size

        self.q_history = deque(maxlen=history_size)
        self.visit_counts = torch.zeros(
            action_dim, num_bins, device=device, dtype=torch.float32
        )
        self.cumulative_q = torch.zeros(
            action_dim, num_bins, device=device, dtype=torch.float32
        )
        self.recent_visit_window = deque(maxlen=50)

    def update(self, q_values, actions, active_masks):
        """Update tracking with new Q-values and actions."""
        self.q_history.append(q_values.detach())
        self._update_visit_counts(actions, active_masks)
        self._update_cumulative_q(q_values, actions, active_masks)

    def _update_visit_counts(self, actions, active_masks):
        """Update count of visits to each bin."""
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        visit_snapshot = torch.zeros_like(self.visit_counts)

        for dim in range(self.action_dim):
            active_indices = torch.where(active_masks[dim])[0]
            for action_idx in actions[:, dim]:
                if 0 <= action_idx < len(active_indices):
                    actual_bin = active_indices[action_idx]
                    self.visit_counts[dim, actual_bin] += 1
                    visit_snapshot[dim, actual_bin] += 1

        self.recent_visit_window.append(visit_snapshot)

    def _update_cumulative_q(self, q_values, actions, active_masks):
        """Update cumulative Q-values for visited bins."""
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        batch_size = q_values.shape[0]

        for b in range(batch_size):
            for dim in range(self.action_dim):
                active_indices = torch.where(active_masks[dim])[0]
                action_idx = actions[b, dim]

                if 0 <= action_idx < len(active_indices):
                    actual_bin = active_indices[action_idx]
                    self.cumulative_q[dim, actual_bin] += q_values[b, dim, actual_bin]

    def has_sufficient_data(self):
        """Check if enough data has been collected."""
        return len(self.q_history) >= 20 and self.visit_counts.sum() > 50

    def is_confident(self):
        """Check if Q-value estimates are confident enough for weighted selection."""
        if len(self.q_history) < 30:
            return False

        recent_q = torch.stack(list(self.q_history)[-30:], dim=0)
        q_std = recent_q.std(dim=0).mean()

        return q_std < 15.0

    def get_mean_q_values(self):
        """Compute mean Q-value per bin."""
        mean_q = torch.zeros_like(self.cumulative_q)

        for dim in range(self.action_dim):
            for bin_idx in range(self.num_bins):
                if self.visit_counts[dim, bin_idx] > 0:
                    mean_q[dim, bin_idx] = (
                            self.cumulative_q[dim, bin_idx] /
                            self.visit_counts[dim, bin_idx]
                    )

        return mean_q

    def get_recent_visit_counts(self):
        """Get visit counts from recent window."""
        if len(self.recent_visit_window) == 0:
            return torch.zeros_like(self.visit_counts)

        recent_visits = torch.stack(list(self.recent_visit_window), dim=0)
        return recent_visits.sum(dim=0)