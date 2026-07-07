from __future__ import annotations

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

    env0 = SearchTrackBelief20Env(EnvConfig(grid_size=args.grid_size, seed=args.seed))
    device = pick_device()
    agent = BDQNAgent(BDQNConfig(obs_shape=env0.observation_shape, action_dim=env0.action_dim, device=device, seed=args.seed))
    agent.load(args.checkpoint)

    rewards = []
    detected = []
    completed = []
    known = []
    visited = []
    search = 0
    track = 0
    decisions = 0
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
        total = 0.0
        while not done:
            action_mask = env.action_mask()
            action = agent.act(obs, use_sample=False, action_mask=action_mask)
            search += int(action == 0)
            track += int(action == 1)
            decisions += 1
            obs, r, terminated, truncated, info = env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
        detected.append(info["detected"])
        completed.append(info["completed"])
        known.append(info["known_targets"])
        visited.append(info["visited_ratio"])
    print({
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "detected_mean": float(np.mean(detected)),
        "completed_mean": float(np.mean(completed)),
        "known_targets_mean": float(np.mean(known)),
        "visited_ratio_mean": float(np.mean(visited)),
        "search_ratio": search / max(1, decisions),
        "track_ratio": track / max(1, decisions),
    })


if __name__ == "__main__":
    main()
