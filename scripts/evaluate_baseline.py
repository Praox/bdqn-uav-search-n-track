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

    detected_value = []
    completed_value = []

    detected_value1 = []
    detected_value2 = []

    completed_value1 = []
    completed_value2 = []
    
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
            detect_value1_bonus=args.detect_value1_bonus,
            detect_value2_bonus=args.detect_value2_bonus,
            track_progress_value1_bonus=args.track_progress_value1_bonus,
            track_progress_value2_bonus=args.track_progress_value2_bonus,
            complete_value1_bonus=args.complete_value1_bonus,
            complete_value2_bonus=args.complete_value2_bonus,
            track_step_penalty=args.track_step_penalty,
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
        detected_value.append(info["detected_value"])
        completed_value.append(info["completed_value"])

        detected_value1.append(info["detected_value1"])
        detected_value2.append(info["detected_value2"])

        completed_value1.append(info["completed_value1"])
        completed_value2.append(info["completed_value2"])

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
        
        "detected_value_mean": safe_mean(detected_value),
        "completed_value_mean": safe_mean(completed_value),

        "detected_value1_mean": safe_mean(detected_value1),
        "detected_value2_mean": safe_mean(detected_value2),

        "completed_value1_mean": safe_mean(completed_value1),
        "completed_value2_mean": safe_mean(completed_value2),

        "value2_detection_rate": safe_mean(detected_value2) / max(1, args.n_value2_targets),
        "value2_completion_rate": safe_mean(completed_value2) / max(1, args.n_value2_targets),

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
    p.add_argument("--run-dir", type=str, default="runs")
    
    p.add_argument("--detect-value1-bonus", type=float, default=0.30)
    p.add_argument("--detect-value2-bonus", type=float, default=1.00)

    p.add_argument("--track-progress-value1-bonus", type=float, default=0.03)
    p.add_argument("--track-progress-value2-bonus", type=float, default=0.12)

    p.add_argument("--complete-value1-bonus", type=float, default=2.00)
    p.add_argument("--complete-value2-bonus", type=float, default=8.00)

    p.add_argument("--track-step-penalty", type=float, default=-0.02)
        
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