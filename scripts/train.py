from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import trange

from uav_bdqn_belief20.envs.search_track_env import SearchTrackBelief20Env, EnvConfig
from uav_bdqn_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(agent: BDQNAgent, args, episodes: int = 20) -> dict:
    vals = []
    detected = []
    completed = []
    known = []
    coverage = []

    search = 0
    track = 0
    decisions = 0

    available_track_decisions = 0
    track_when_available = 0
    search_when_available = 0

    for ep in range(episodes):
        env = SearchTrackBelief20Env(EnvConfig(
            grid_size=args.grid_size,
            n_value1_targets=args.n_value1_targets,
            n_value2_targets=args.n_value2_targets,
            sensor_radius=args.sensor_radius,
            detection_probability=args.detection_probability,
            macro_steps=args.macro_steps,
            max_steps=args.max_steps,
            seed=args.seed + 10_000 + ep,
        ))

        obs, info = env.reset()
        done = False
        total = 0.0

        while not done:
            has_trackable = any(
                not target.completed
                for target in env.memory.known_targets.values()
            )

            if has_trackable:
                available_track_decisions += 1

            action_mask = env.action_mask()
            a = agent.act(obs, use_sample=False, action_mask=action_mask)

            if a == 0:
                search += 1
                if has_trackable:
                    search_when_available += 1
            else:
                track += 1
                if has_trackable:
                    track_when_available += 1

            decisions += 1

            obs, r, terminated, truncated, info = env.step(a)
            total += r
            done = terminated or truncated

        vals.append(total)
        detected.append(info["detected"])
        completed.append(info["completed"])
        known.append(info["known_targets"])
        coverage.append(float(env.memory.visited.mean()))

    return {
        "eval_reward": float(np.mean(vals)),
        "eval_detected": float(np.mean(detected)),
        "eval_completed": float(np.mean(completed)),
        "eval_known_targets": float(np.mean(known)),
        "eval_sensor_coverage": float(np.mean(coverage)),
        "search_ratio": search / max(1, decisions),
        "track_ratio": track / max(1, decisions),
        "track_available_decisions": int(available_track_decisions),
        "track_when_available_ratio": track_when_available / max(1, available_track_decisions),
        "search_when_available_ratio": search_when_available / max(1, available_track_decisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--macro-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print(f"Using device: {device}")

    env = SearchTrackBelief20Env(EnvConfig(
        grid_size=args.grid_size,
        n_value1_targets=args.n_value1_targets,
        n_value2_targets=args.n_value2_targets,
        sensor_radius=args.sensor_radius,
        detection_probability=args.detection_probability,
        macro_steps=args.macro_steps,
        max_steps=args.max_steps,
        seed=args.seed,
    ))

    agent = BDQNAgent(BDQNConfig(
        obs_shape=env.observation_shape,
        action_dim=env.action_dim,
        device=device,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
    ))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    recent_reward = deque(maxlen=50)
    recent_detected = deque(maxlen=50)
    recent_completed = deque(maxlen=50)
    recent_search = deque(maxlen=200)
    recent_track = deque(maxlen=200)

    # IMPORTANT:
    # This must be outside the training loop.
    # Otherwise it resets every episode and best.pt becomes meaningless.
    best_score = -1e9
    best_ep = 0

    pbar = trange(args.episodes, desc="BDQN belief20")

    for ep in pbar:
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action_mask = env.action_mask()
            action = agent.act(obs, use_sample=True, action_mask=action_mask)

            recent_search.append(1 if action == 0 else 0)
            recent_track.append(1 if action == 1 else 0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay.add(obs, action, reward, next_obs, done)
            agent.train_step()

            obs = next_obs
            ep_reward += reward

        recent_reward.append(ep_reward)
        recent_detected.append(info["detected"])
        recent_completed.append(info["completed"])

        pbar.set_postfix(
            reward=f"{np.mean(recent_reward):.2f}",
            det=f"{np.mean(recent_detected):.2f}",
            comp=f"{np.mean(recent_completed):.2f}",
            search=f"{np.mean(recent_search):.2f}",
            track=f"{np.mean(recent_track):.2f}",
            best=f"{best_score:.2f}",
        )

        if (ep + 1) % 100 == 0:
            metrics = evaluate(agent, args, episodes=args.eval_episodes)
            print(f"\n[Eval {ep + 1}] {metrics}")

            # Always save latest.
            agent.save(str(run_dir / "latest.pt"))

            # Score used to choose best checkpoint.
            # Completion is weighted because the task is not only detection.
            score = (
                metrics["eval_reward"]
                + 2.0 * metrics["eval_completed"]
                + 1.0 * metrics["eval_detected"]
            )

            if score > best_score:
                best_score = score
                best_ep = ep + 1
                agent.save(str(run_dir / "best.pt"))
                print(f"[Best] saved best.pt at episode {best_ep} with score={best_score:.3f}")

    agent.save(str(run_dir / "latest.pt"))

    print("Training complete.")
    print(f"Best checkpoint: runs/best.pt from episode {best_ep} with score={best_score:.3f}")


if __name__ == "__main__":
    main()