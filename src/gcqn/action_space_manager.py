import torch
from collections import deque


class CQNActionSpaceManager:
    """
    Manages growing action space with adaptive unmasking.
    Combines GQN's preservation through masking with CQN's coarse-to-fine philosophy.
    """

    def __init__(self, action_spec, initial_bins, final_bins, device):
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

        self.action_bins = self._create_full_action_grid()
        self.active_masks = self._initialize_coarse_masks()
        self.growth_history = []

        self.metrics_tracker = UnmaskingMetricsTracker(
            self.action_dim, self.final_bins, device
        )

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

    def _initialize_coarse_masks(self):
        """Initialize masks with only coarse bins active."""
        masks = torch.zeros(
            self.action_dim, self.final_bins, dtype=torch.bool, device=self.device
        )

        coarse_indices = self._compute_coarse_bin_indices()
        for dim in range(self.action_dim):
            masks[dim, coarse_indices] = True

        return masks

    def _compute_coarse_bin_indices(self):
        """Compute evenly-spaced coarse bin indices."""
        if self.initial_bins == self.final_bins:
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
        """Update metrics for adaptive unmasking decisions."""
        self.metrics_tracker.update(q_values, actions, self.active_masks)

    def check_and_unmask(self, episode, unmasking_strategy="variance"):
        """Check if bins should be unmasked and perform unmasking."""
        if not self._can_unmask():
            return False

        metrics = self.metrics_tracker.compute_metrics()
        bins_to_unmask = self._identify_bins_to_unmask(metrics, unmasking_strategy)

        if len(bins_to_unmask) == 0:
            return False

        self._unmask_bins(bins_to_unmask, episode)
        return True

    def _can_unmask(self):
        """Check if any dimensions have inactive bins remaining."""
        return not torch.all(self.active_masks)

    def _identify_bins_to_unmask(self, metrics, strategy):
        """Identify which bins should be unmasked based on metrics."""
        if strategy == "variance":
            return self._variance_based_unmasking(metrics)
        elif strategy == "advantage":
            return self._advantage_based_unmasking(metrics)
        elif strategy == "hybrid":
            return self._hybrid_metric_unmasking(metrics)
        else:
            raise ValueError(f"Unknown unmasking strategy: {strategy}")

    def _variance_based_unmasking(self, metrics):
        """Unmask bins with high Q-value variance."""
        variances = metrics["q_variance"]
        threshold = torch.quantile(variances[self.active_masks], 0.75)

        high_variance_bins = (variances > threshold) & self.active_masks
        return self._get_neighbors_to_unmask(high_variance_bins)

    def _advantage_based_unmasking(self, metrics):
        """Unmask bins with high advantage values."""
        advantages = metrics["q_advantage"]
        threshold = torch.quantile(advantages[self.active_masks], 0.75)

        high_advantage_bins = (advantages > threshold) & self.active_masks
        return self._get_neighbors_to_unmask(high_advantage_bins)

    def _hybrid_metric_unmasking(self, metrics):
        """Unmask bins using weighted combination of metrics."""
        variance_norm = self._normalize_metric(metrics["q_variance"])
        advantage_norm = self._normalize_metric(metrics["q_advantage"])
        visit_norm = self._normalize_metric(metrics["visit_counts"])

        hybrid_score = 0.4 * variance_norm + 0.4 * advantage_norm + 0.2 * visit_norm
        threshold = torch.quantile(hybrid_score[self.active_masks], 0.75)

        high_score_bins = (hybrid_score > threshold) & self.active_masks
        return self._get_neighbors_to_unmask(high_score_bins)

    def _normalize_metric(self, metric):
        """Normalize metric to [0, 1] range."""
        min_val = metric.min()
        max_val = metric.max()
        if max_val - min_val < 1e-8:
            return torch.zeros_like(metric)
        return (metric - min_val) / (max_val - min_val)

    def _get_neighbors_to_unmask(self, seed_bins):
        """Get neighboring bins that should be unmasked."""
        neighbors_to_unmask = []

        for dim in range(self.action_dim):
            active_bin_indices = torch.where(seed_bins[dim])[0]

            for bin_idx in active_bin_indices:
                left_neighbor = bin_idx - 1
                right_neighbor = bin_idx + 1

                if left_neighbor >= 0 and not self.active_masks[dim, left_neighbor]:
                    neighbors_to_unmask.append((dim, left_neighbor.item()))

                if (right_neighbor < self.final_bins and
                        not self.active_masks[dim, right_neighbor]):
                    neighbors_to_unmask.append((dim, right_neighbor.item()))

        return neighbors_to_unmask

    def _unmask_bins(self, bins_to_unmask, episode):
        """Unmask specified bins and record growth event."""
        for dim, bin_idx in bins_to_unmask:
            self.active_masks[dim, bin_idx] = True

        self.growth_history.append({
            "episode": episode,
            "bins_unmasked": bins_to_unmask,
            "total_active": self.active_masks.sum().item()
        })

    def get_growth_info(self):
        """Get information about current growth state."""
        return {
            "total_active_bins": self.active_masks.sum().item(),
            "total_possible_bins": self.action_dim * self.final_bins,
            "active_per_dimension": [
                self.active_masks[d].sum().item()
                for d in range(self.action_dim)
            ],
            "growth_events": len(self.growth_history)
        }


class UnmaskingMetricsTracker:
    """Tracks metrics for adaptive unmasking decisions."""

    def __init__(self, action_dim, num_bins, device, history_size=100):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device
        self.history_size = history_size

        self.q_history = deque(maxlen=history_size)
        self.visit_counts = torch.zeros(
            action_dim, num_bins, device=device, dtype=torch.float32
        )

    def update(self, q_values, actions, active_masks):
        """Update tracking with new Q-values and actions."""
        self.q_history.append(q_values.detach())
        self._update_visit_counts(actions, active_masks)

    def _update_visit_counts(self, actions, active_masks):
        """Update count of visits to each bin."""
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        for dim in range(self.action_dim):
            active_indices = torch.where(active_masks[dim])[0]
            for action_idx in actions[:, dim]:
                if 0 <= action_idx < len(active_indices):
                    actual_bin = active_indices[action_idx]
                    self.visit_counts[dim, actual_bin] += 1

    def compute_metrics(self):
        """Compute metrics for unmasking decisions."""
        if len(self.q_history) < 10:
            return self._default_metrics()

        stacked_q = torch.stack(list(self.q_history), dim=0)

        return {
            "q_variance": self._compute_variance(stacked_q),
            "q_advantage": self._compute_advantage(stacked_q),
            "visit_counts": self.visit_counts
        }

    def _compute_variance(self, stacked_q):
        """Compute variance of Q-values over history."""
        return stacked_q.var(dim=0).mean(dim=0)

    def _compute_advantage(self, stacked_q):
        """Compute advantage of each bin relative to mean."""
        q_mean = stacked_q.mean(dim=[0, 2])
        recent_q = stacked_q[-10:].mean(dim=0)

        advantages = torch.zeros(
            self.action_dim, self.num_bins, device=self.device
        )
        for dim in range(self.action_dim):
            advantages[dim] = recent_q[dim] - q_mean[dim]

        return advantages

    def _default_metrics(self):
        """Return default metrics when insufficient history."""
        return {
            "q_variance": torch.zeros(
                self.action_dim, self.num_bins, device=self.device
            ),
            "q_advantage": torch.zeros(
                self.action_dim, self.num_bins, device=self.device
            ),
            "visit_counts": self.visit_counts
        }