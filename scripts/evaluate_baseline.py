from __future__ import annotations

import argparse
import numpy as np

from uav_bdqn_belief20.envs.search_track_env import SearchTrackBelief20Env, EnvConfig


SEARCH = 0
TRACK = 1


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) > 0 else 0.0


def evaluate_policy(policy_name: str, args) -> dict:
    rewards = []
    detected = []
    completed = []
    known = []
    coverage = []
    new_observed_cells_per_decision = []

    search = 0
    track = 0
    decisions = 0

    available_track_decisions = 0
    track_when_available = 0
    search_when_available = 0

    first_detection_steps = []
    first_completion_steps = []

    n_total_targets = args.n_value1_targets + args.n_value2_targets

    rng = np.random.default_rng(args.seed)

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
            prev_observed_count = int(env.memory.visited.sum())

            has_trackable = any(
                not target.completed
                for target in env.memory.known_targets.values()
            )

            if has_trackable:
                available_track_decisions += 1

            if policy_name == "search_only":
                action = SEARCH

            elif policy_name == "track_if_available":
                action = TRACK if has_trackable else SEARCH

            elif policy_name == "random_valid":
                mask = env.action_mask()
                valid_actions = np.flatnonzero(mask)
                action = int(rng.choice(valid_actions))

            else:
                raise ValueError(f"Unknown policy_name: {policy_name}")

            if action == SEARCH:
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

            new_observed_count = int(env.memory.visited.sum())
            newly_observed = max(0, new_observed_count - prev_observed_count)
            ep_new_observed_cells += newly_observed

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
        coverage.append(float(env.memory.visited.mean()))

        new_observed_cells_per_decision.append(
            ep_new_observed_cells / max(1, ep_decisions)
        )

        first_detection_steps.append(
            ep_first_detection_step if ep_first_detection_step is not None else args.max_steps
        )

        first_completion_steps.append(
            ep_first_completion_step if ep_first_completion_step is not None else args.max_steps
        )

    return {
        "policy": policy_name,

        "reward_mean": safe_mean(rewards),

        "detected_mean": safe_mean(detected),
        "detection_rate_mean": safe_mean([x / max(1, n_total_targets) for x in detected]),

        "completed_mean": safe_mean(completed),
        "completion_rate_mean": safe_mean([x / max(1, n_total_targets) for x in completed]),

        "known_targets_mean": safe_mean(known),

        "track_success_rate_mean": safe_mean([
            c / max(1, k)
            for c, k in zip(completed, known)
        ]),

        "sensor_coverage_ratio_mean": safe_mean(coverage),
        "new_observed_cells_per_decision_mean": safe_mean(new_observed_cells_per_decision),

        "search_ratio": search / max(1, decisions),
        "track_ratio": track / max(1, decisions),

        "track_available_decisions": int(available_track_decisions),
        "track_when_available_ratio": track_when_available / max(1, available_track_decisions),
        "search_when_available_ratio": search_when_available / max(1, available_track_decisions),

        "first_detection_step_mean": safe_mean(first_detection_steps),
        "first_completion_step_mean": safe_mean(first_completion_steps),
    }


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--policy", type=str, default="all",
                   choices=["all", "search_only", "track_if_available", "random_valid"])

    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--n-value1-targets", type=int, default=3)
    p.add_argument("--n-value2-targets", type=int, default=1)
    p.add_argument("--sensor-radius", type=int, default=2)
    p.add_argument("--detection-probability", type=float, default=1.0)
    p.add_argument("--macro-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=123)

    args = p.parse_args()

    policies = (
        ["search_only", "track_if_available", "random_valid"]
        if args.policy == "all"
        else [args.policy]
    )

    for policy_name in policies:
        metrics = evaluate_policy(policy_name, args)
        print(metrics)


if __name__ == "__main__":
    main()