from __future__ import annotations
from pprint import pprint

import argparse
import numpy as np
import torch

from uav_bdqn_belief20.envs.search_track_env import SearchTrackBelief20Env, EnvConfig
from uav_bdqn_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) > 0 else 0.0


def safe_std(xs):
    return float(np.std(xs)) if len(xs) > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="runs/latest.pt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--n-value1-targets", type=int, default=3)
    p.add_argument("--n-value2-targets", type=int, default=1)
    p.add_argument("--sensor-radius", type=int, default=2)
    p.add_argument("--detection-probability", type=float, default=1.0)
    p.add_argument("--macro-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    env0 = SearchTrackBelief20Env(EnvConfig(
        grid_size=args.grid_size,
        n_value1_targets=args.n_value1_targets,
        n_value2_targets=args.n_value2_targets,
        sensor_radius=args.sensor_radius,
        detection_probability=args.detection_probability,
        macro_steps=args.macro_steps,
        max_steps=args.max_steps,
        seed=args.seed,
    ))

    device = pick_device()
    agent = BDQNAgent(BDQNConfig(
        obs_shape=env0.observation_shape,
        action_dim=env0.action_dim,
        device=device,
        seed=args.seed,
    ))
    agent.load(args.checkpoint)

    rewards = []
    detected = []
    completed = []
    known = []
    visited_ratio = []

    search = 0
    track = 0
    decisions = 0

    # Conditional mission metrics
    available_track_decisions = 0
    track_when_available = 0
    search_when_available = 0

    # Exploration efficiency
    new_observed_cells_per_decision = []
    total_new_observed_cells = 0

    # Timing metrics
    first_detection_steps = []
    first_completion_steps = []

    # Success metrics
    track_success_rates = []
    detection_rates = []
    completion_rates = []

    n_total_targets = args.n_value1_targets + args.n_value2_targets

    for ep in range(args.episodes):
        env = SearchTrackBelief20Env(EnvConfig(
            grid_size=args.grid_size,
            n_value1_targets=args.n_value1_targets,
            n_value2_targets=args.n_value2_targets,
            sensor_radius=args.sensor_radius,
            detection_probability=args.detection_probability,
            macro_steps=args.macro_steps,
            max_steps=args.max_steps,
            seed=args.seed + ep,
        ))

        obs, info = env.reset()
        done = False
        total_reward = 0.0

        ep_first_detection_step = None
        ep_first_completion_step = None

        ep_new_observed_cells = 0
        ep_decisions = 0

        while not done:
            prev_detected = info["detected"]
            prev_completed = info["completed"]

            # Coverage before action
            prev_observed_count = int(env.memory.visited.sum())

            has_trackable = any(
                not target.completed
                for target in env.memory.known_targets.values()
            )

            if has_trackable:
                available_track_decisions += 1

            action_mask = env.action_mask()
            action = agent.act(obs, use_sample=False, action_mask=action_mask)

            if action == 0:
                search += 1
                if has_trackable:
                    search_when_available += 1
            else:
                track += 1
                if has_trackable:
                    track_when_available += 1

            decisions += 1
            ep_decisions += 1

            obs, reward, terminated, truncated, info = env.step(action)

            # Coverage after action
            new_observed_count = int(env.memory.visited.sum())
            newly_observed = max(0, new_observed_count - prev_observed_count)
            ep_new_observed_cells += newly_observed
            total_new_observed_cells += newly_observed

            total_reward += reward
            done = terminated or truncated

            if ep_first_detection_step is None and info["detected"] > prev_detected:
                ep_first_detection_step = info["t"]

            if ep_first_completion_step is None and info["completed"] > prev_completed:
                ep_first_completion_step = info["t"]

        rewards.append(total_reward)
        detected.append(info["detected"])
        completed.append(info["completed"])
        known.append(info["known_targets"])
        visited_ratio.append(float(env.memory.visited.mean()))

        first_detection_steps.append(
            ep_first_detection_step if ep_first_detection_step is not None else args.max_steps
        )

        first_completion_steps.append(
            ep_first_completion_step if ep_first_completion_step is not None else args.max_steps
        )

        new_observed_cells_per_decision.append(
            ep_new_observed_cells / max(1, ep_decisions)
        )

        detection_rates.append(
            info["detected"] / max(1, n_total_targets)
        )

        completion_rates.append(
            info["completed"] / max(1, n_total_targets)
        )

        track_success_rates.append(
            info["completed"] / max(1, info["known_targets"])
        )

    metrics = {
        # Main performance
        "reward_mean": safe_mean(rewards),
        #"reward_std": safe_std(rewards),
        
        # Search / detection performance
        "detected_mean": safe_mean(detected),
        #"detection_rate_mean": safe_mean(detection_rates),
        "known_targets_mean": safe_mean(known),

        # Track / completion performance
        "completed_mean": safe_mean(completed),
        "completion_rate_mean": safe_mean(completion_rates),
        "track_success_rate_mean": safe_mean(track_success_rates),

        # Coverage / exploration
        "sensor_coverage_ratio_mean": safe_mean(visited_ratio),
        "new_observed_cells_per_decision_mean": safe_mean(new_observed_cells_per_decision),

        # Global action ratios
        "search_ratio": search / max(1, decisions),
        "track_ratio": track / max(1, decisions),

        # Conditional action ratios
        "track_available_decisions": int(available_track_decisions),
        "track_when_available_ratio": track_when_available / max(1, available_track_decisions),
        "search_when_available_ratio": search_when_available / max(1, available_track_decisions),

        # Timing
        #"first_detection_step_mean": safe_mean(first_detection_steps),
        #"first_completion_step_mean": safe_mean(first_completion_steps),
    }

    pprint(metrics)


if __name__ == "__main__":
    main()