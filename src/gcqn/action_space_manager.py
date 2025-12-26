import torch
from collections import deque


class GCQNActionSpaceManager:
    """
    True coarse-to-fine action space manager.
    Phase 1: Wide coverage with uniform coarse bins
    Phase 2: Identify important regions through metrics
    Phase 3: Prune low-value bins, refine high-value regions
    Result: Fewer bins, but better positioned (adaptive resolution)
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
        self.active_masks = self._initialize_wide_coverage_masks()
        self.growth_history = []
        self.pruning_history = []
        self.current_phase = 1

        self.metrics_tracker = BinImportanceTracker(
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

    def _initialize_wide_coverage_masks(self):
        """Initialize with wide coverage - uniform coarse bins."""
        masks = torch.zeros(
            self.action_dim, self.final_bins, dtype=torch.bool, device=self.device
        )

        coarse_indices = self._compute_uniform_coarse_indices()
        for dim in range(self.action_dim):
            masks[dim, coarse_indices] = True

        return masks

    def _compute_uniform_coarse_indices(self):
        """Compute evenly-spaced indices for wide coverage."""
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
            torch.where(self.active_masks[dim])[0] for dim in range(self.action_dim)
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
        """Update metrics for adaptive pruning/refinement decisions."""
        self.metrics_tracker.update(q_values, actions, self.active_masks)

    def check_and_adapt(self, episode, min_episodes_phase2, min_episodes_phase3):
        """
        Check if action space should be adapted.
        Returns tuple: (adapted, phase_changed)
        """
        if episode < min_episodes_phase2:
            return False, False

        if self.current_phase == 1 and episode >= min_episodes_phase2:
            self.current_phase = 2
            return False, True

        if self.current_phase == 2 and episode >= min_episodes_phase3:
            self.current_phase = 3
            adapted = self._perform_pruning_and_refinement(episode)
            return adapted, True

        if self.current_phase == 3:
            adapted = self._perform_selective_refinement(episode)
            return adapted, False

        return False, False

    def _perform_pruning_and_refinement(self, episode):
        """
        Phase 3: Prune low-value bins and refine high-value regions.
        This is the core of true coarse-to-fine.
        """
        if not self._has_sufficient_metrics():
            return False

        metrics = self.metrics_tracker.compute_metrics()
        importance = self._compute_bin_importance(metrics)

        bins_pruned = self._prune_low_importance_bins(importance, episode)
        bins_refined = self._refine_high_importance_bins(importance, episode)

        return bins_pruned > 0 or bins_refined > 0

    def _perform_selective_refinement(self, episode):
        """
        Ongoing refinement: Add detail to promising regions without pruning.
        """
        if not self._has_sufficient_metrics():
            return False

        metrics = self.metrics_tracker.compute_metrics()
        importance = self._compute_bin_importance(metrics)

        bins_refined = self._refine_high_importance_bins(importance, episode)
        return bins_refined > 0

    def _has_sufficient_metrics(self):
        """Check if we have enough data for reliable decisions."""
        return len(self.metrics_tracker.q_history) >= 20

    def _compute_bin_importance(self, metrics):
        """
        Compute importance score for each bin.
        High importance = high Q-value + high visits + high variance
        """
        q_variance_norm = self._normalize_metric(metrics["q_variance"])
        q_advantage_norm = self._normalize_metric(metrics["q_advantage"])
        visit_norm = self._normalize_metric(metrics["visit_counts"])

        importance = 0.3 * q_variance_norm + 0.5 * q_advantage_norm + 0.2 * visit_norm
        return importance

    def _normalize_metric(self, metric):
        """Normalize metric to [0, 1] range."""
        min_val = metric.min()
        max_val = metric.max()
        if max_val - min_val < 1e-8:
            return torch.zeros_like(metric)
        return (metric - min_val) / (max_val - min_val)

    def _prune_low_importance_bins(self, importance, episode):
        """
        Prune bins with low importance (deactivate them).
        This is what makes it true coarse-to-fine: we REDUCE bins.
        """
        bins_pruned = 0
        prune_threshold = 0.2

        for dim in range(self.action_dim):
            active_bins = torch.where(self.active_masks[dim])[0]

            if len(active_bins) <= 2:
                continue

            for bin_idx in active_bins:
                if importance[dim, bin_idx] < prune_threshold:
                    if self._can_prune_bin(dim, bin_idx):
                        self.active_masks[dim, bin_idx] = False
                        bins_pruned += 1

        if bins_pruned > 0:
            self.pruning_history.append({
                "episode": episode,
                "bins_pruned": bins_pruned,
                "total_active": self.active_masks.sum().item()
            })

        return bins_pruned

    def _can_prune_bin(self, dim, bin_idx):
        """
        Check if bin can be safely pruned.
        Don't prune if it would isolate important regions.
        """
        active_bins = torch.where(self.active_masks[dim])[0]

        temp_mask = self.active_masks[dim].clone()
        temp_mask[bin_idx] = False
        remaining_active = torch.where(temp_mask)[0]

        if len(remaining_active) < 2:
            return False

        gaps = remaining_active[1:] - remaining_active[:-1]
        max_gap = gaps.max().item() if len(gaps) > 0 else 0

        return max_gap <= 3

    def _refine_high_importance_bins(self, importance, episode):
        """
        Add neighboring bins to high-importance regions for finer control.
        """
        bins_refined = 0
        refine_threshold = 0.7

        high_importance_bins = importance > refine_threshold

        for dim in range(self.action_dim):
            active_high_bins = high_importance_bins[dim] & self.active_masks[dim]
            high_bin_indices = torch.where(active_high_bins)[0]

            for bin_idx in high_bin_indices:
                left_neighbor = bin_idx - 1
                right_neighbor = bin_idx + 1

                if left_neighbor >= 0 and not self.active_masks[dim, left_neighbor]:
                    self.active_masks[dim, left_neighbor] = True
                    bins_refined += 1

                if (right_neighbor < self.final_bins and
                        not self.active_masks[dim, right_neighbor]):
                    self.active_masks[dim, right_neighbor] = True
                    bins_refined += 1

        if bins_refined > 0:
            self.growth_history.append({
                "episode": episode,
                "bins_refined": bins_refined,
                "total_active": self.active_masks.sum().item()
            })

        return bins_refined

    def get_growth_info(self):
        """Get information about current adaptation state."""
        return {
            "current_phase": self.current_phase,
            "total_active_bins": self.active_masks.sum().item(),
            "total_possible_bins": self.action_dim * self.final_bins,
            "active_per_dimension": [
                self.active_masks[d].sum().item() for d in range(self.action_dim)
            ],
            "refinement_events": len(self.growth_history),
            "pruning_events": len(self.pruning_history)
        }

    def get_visual_representation(self, logger=None):
        """Generate visual representation of active bins per dimension."""
        visualization = []

        for dim in range(self.action_dim):
            active_indices = torch.where(self.active_masks[dim])[0].cpu().numpy()
            dim_viz = self._create_dimension_visualization(dim, active_indices)
            visualization.append(dim_viz)

            if logger:
                logger.info(f"  Dim {dim}: {dim_viz['ascii']}")
                logger.info(
                    f"         Active: {dim_viz['active_bins']}/{self.final_bins} | "
                    f"Range: [{dim_viz['min_val']:.2f}, {dim_viz['max_val']:.2f}]"
                )

        return visualization

    def _create_dimension_visualization(self, dim, active_indices):
        """Create visualization for single dimension."""
        ascii_repr = []

        for bin_idx in range(self.final_bins):
            if bin_idx in active_indices:
                ascii_repr.append("█")
            else:
                ascii_repr.append("░")

        active_bins_values = self.action_bins[dim, active_indices].cpu().numpy()

        return {
            "dimension": dim,
            "ascii": "".join(ascii_repr),
            "active_bins": len(active_indices),
            "min_val": active_bins_values.min() if len(active_bins_values) > 0 else 0,
            "max_val": active_bins_values.max() if len(active_bins_values) > 0 else 0,
            "active_indices": active_indices.tolist()
        }

    def get_importance_visualization(self, logger=None):
        """Visualize bin importance scores."""
        if not self._has_sufficient_metrics():
            return None

        metrics = self.metrics_tracker.compute_metrics()
        importance = self._compute_bin_importance(metrics)

        visualization = []

        for dim in range(self.action_dim):
            dim_viz = self._create_importance_visualization(dim, importance[dim])
            visualization.append(dim_viz)

            if logger:
                logger.info(f"  Dim {dim}: {dim_viz['ascii']}")
                logger.info(
                    f"         Avg importance: {dim_viz['avg_importance']:.3f} | "
                    f"High-value bins: {dim_viz['high_value_count']}"
                )

        return visualization

    def _create_importance_visualization(self, dim, importance_scores):
        """Create importance visualization for single dimension."""
        ascii_repr = []
        high_value_threshold = 0.7
        high_value_count = 0

        for bin_idx in range(self.final_bins):
            if self.active_masks[dim, bin_idx]:
                score = importance_scores[bin_idx].item()
                if score > high_value_threshold:
                    ascii_repr.append("▓")
                    high_value_count += 1
                elif score > 0.3:
                    ascii_repr.append("█")
                else:
                    ascii_repr.append("▒")
            else:
                ascii_repr.append("░")

        active_importance = importance_scores[self.active_masks[dim]]
        avg_importance = active_importance.mean().item() if len(active_importance) > 0 else 0

        return {
            "dimension": dim,
            "ascii": "".join(ascii_repr),
            "avg_importance": avg_importance,
            "high_value_count": high_value_count
        }

    def log_detailed_state(self, logger, episode):
        """Log comprehensive state information."""
        logger.info("=" * 80)
        logger.info(f"TRUE COARSE-TO-FINE STATE at Episode {episode}")
        logger.info("=" * 80)

        growth_info = self.get_growth_info()
        phase_names = {1: "Wide Coverage", 2: "Learning Importance", 3: "Pruning & Refinement"}
        logger.info(f"Current Phase: {growth_info['current_phase']} - {phase_names[growth_info['current_phase']]}")
        logger.info(
            f"Active bins: {growth_info['total_active_bins']}/{growth_info['total_possible_bins']} "
            f"({100 * growth_info['total_active_bins'] / growth_info['total_possible_bins']:.1f}%)"
        )
        logger.info(f"Refinement events: {growth_info['refinement_events']}")
        logger.info(f"Pruning events: {growth_info['pruning_events']}")
        logger.info("")

        logger.info("Active Bins per Dimension:")
        logger.info("Legend: █ = Active  ░ = Inactive  ▓ = High Importance  ▒ = Low Importance")
        self.get_visual_representation(logger)
        logger.info("")

        if self._has_sufficient_metrics():
            logger.info("Bin Importance Scores:")
            self.get_importance_visualization(logger)

        logger.info("=" * 80)


class BinImportanceTracker:
    """Tracks metrics for determining bin importance."""

    def __init__(self, action_dim, num_bins, device, history_size=100):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device
        self.history_size = history_size

        self.q_history = deque(maxlen=history_size)
        self.visit_counts = torch.zeros(
            action_dim, num_bins, device=device, dtype=torch.float32
        )
        self.q_value_sums = torch.zeros(
            action_dim, num_bins, device=device, dtype=torch.float32
        )

    def update(self, q_values, actions, active_masks):
        """Update tracking with new Q-values and actions."""
        self.q_history.append(q_values.detach())
        self._update_visit_counts(actions, active_masks)
        self._update_q_value_sums(q_values, actions, active_masks)

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

    def _update_q_value_sums(self, q_values, actions, active_masks):
        """Update sum of Q-values for each bin."""
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        batch_size = q_values.shape[0]

        for dim in range(self.action_dim):
            active_indices = torch.where(active_masks[dim])[0]
            for b in range(batch_size):
                action_idx = actions[b, dim]
                if 0 <= action_idx < len(active_indices):
                    actual_bin = active_indices[action_idx]
                    if actual_bin < q_values.shape[2]:
                        self.q_value_sums[dim, actual_bin] += q_values[b, dim, actual_bin]

    def compute_metrics(self):
        """Compute metrics for adaptation decisions."""
        if len(self.q_history) < 10:
            return self._default_metrics()

        stacked_q = torch.stack(list(self.q_history), dim=0)

        return {
            "q_variance": self._compute_variance(stacked_q),
            "q_advantage": self._compute_advantage(stacked_q),
            "visit_counts": self.visit_counts,
            "mean_q_values": self._compute_mean_q_values()
        }

    def _compute_variance(self, stacked_q):
        """Compute variance of Q-values over history."""
        return stacked_q.var(dim=0).mean(dim=0)

    def _compute_advantage(self, stacked_q):
        """Compute advantage of each bin relative to mean."""
        recent_q = stacked_q[-20:].mean(dim=0)

        advantages = torch.zeros(self.action_dim, self.num_bins, device=self.device)

        for dim in range(self.action_dim):
            dim_q_mean = recent_q[:, dim, :].mean()
            advantages[dim] = recent_q[:, dim, :].mean(dim=0) - dim_q_mean

        return advantages

    def _compute_mean_q_values(self):
        """Compute mean Q-values per bin."""
        mean_q = torch.zeros(self.action_dim, self.num_bins, device=self.device)

        for dim in range(self.action_dim):
            for bin_idx in range(self.num_bins):
                if self.visit_counts[dim, bin_idx] > 0:
                    mean_q[dim, bin_idx] = (
                            self.q_value_sums[dim, bin_idx] / self.visit_counts[dim, bin_idx]
                    )

        return mean_q

    def _default_metrics(self):
        """Return default metrics when insufficient history."""
        return {
            "q_variance": torch.zeros(self.action_dim, self.num_bins, device=self.device),
            "q_advantage": torch.zeros(self.action_dim, self.num_bins, device=self.device),
            "visit_counts": self.visit_counts,
            "mean_q_values": torch.zeros(self.action_dim, self.num_bins, device=self.device)
        }