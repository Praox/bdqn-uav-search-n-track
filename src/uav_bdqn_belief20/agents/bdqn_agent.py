from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uav_bdqn_belief20.models.networks import GridFeatureNet
from uav_bdqn_belief20.agents.replay_buffer import ReplayBuffer, Batch


@dataclass
class BDQNConfig:
    obs_shape: tuple[int, int, int] = (5, 20, 20)
    action_dim: int = 2
    feature_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_update_period: int = 500
    posterior_update_period: int = 100
    blr_lambda: float = 1.0
    blr_noise_var: float = 1.0
    device: str = "cpu"
    seed: int = 42


class BayesianLinearHead:
    def __init__(self, action_dim: int, feature_dim: int, lam: float = 1.0, noise_var: float = 1.0, seed: int = 42):
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.lam = float(lam)
        self.noise_var = float(noise_var)
        self.rng = np.random.default_rng(seed)
        self.mu = np.zeros((action_dim, feature_dim), dtype=np.float32)
        self.cov = np.stack([np.eye(feature_dim, dtype=np.float32) / self.lam for _ in range(action_dim)])
        self.sampled_w = self.mu.copy()

    def update(self, phi: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> None:
        for a in range(self.action_dim):
            mask = actions == a
            if not np.any(mask):
                continue
            X = phi[mask].astype(np.float64)
            y = targets[mask].astype(np.float64)
            precision = self.lam * np.eye(self.feature_dim) + (X.T @ X) / self.noise_var
            cov = np.linalg.inv(precision)
            mu = cov @ (X.T @ y) / self.noise_var
            self.cov[a] = cov.astype(np.float32)
            self.mu[a] = mu.astype(np.float32)

    def sample(self) -> None:
        ws = []
        for a in range(self.action_dim):
            cov = self.cov[a] + 1e-6 * np.eye(self.feature_dim, dtype=np.float32)
            ws.append(self.rng.multivariate_normal(self.mu[a], cov).astype(np.float32))
        self.sampled_w = np.stack(ws, axis=0)

    def state_dict(self) -> dict:
        return {
            "mu": torch.as_tensor(self.mu),
            "cov": torch.as_tensor(self.cov),
            "sampled_w": torch.as_tensor(self.sampled_w),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mu = state["mu"].detach().cpu().numpy().astype(np.float32)
        self.cov = state["cov"].detach().cpu().numpy().astype(np.float32)
        self.sampled_w = state["sampled_w"].detach().cpu().numpy().astype(np.float32)


class BDQNAgent:
    def __init__(self, cfg: BDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        c, h, w = cfg.obs_shape
        assert h == w, "this minimal network assumes square map"
        self.feature_net = GridFeatureNet(c, h, cfg.feature_dim).to(self.device)
        self.target_feature_net = GridFeatureNet(c, h, cfg.feature_dim).to(self.device)
        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        self.optim = torch.optim.Adam(self.feature_net.parameters(), lr=cfg.lr)
        self.blr = BayesianLinearHead(cfg.action_dim, cfg.feature_dim, cfg.blr_lambda, cfg.blr_noise_var, seed=cfg.seed)
        self.blr.sample()
        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.obs_shape, seed=cfg.seed)
        self.train_steps = 0

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        use_sample: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        phi = self.feature_net(x).cpu().numpy()[0]

        w = self.blr.sampled_w if use_sample else self.blr.mu
        q = w @ phi

        if action_mask is not None:
            # action_mask example: [True, False]
            # True  = action allowed
            # False = action forbidden
            q = q.copy()
            q[~action_mask] = -1e9

        return int(np.argmax(q))

    def train_step(self) -> dict:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0, "q_mean": 0.0}
        batch = self.replay.sample(self.cfg.batch_size)
        loss, q_mean = self._gradient_update(batch)
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_period == 0:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        if self.train_steps % self.cfg.posterior_update_period == 0:
            self.update_posterior(batch)
            self.blr.sample()
        return {"loss": float(loss), "q_mean": float(q_mean)}

    def _gradient_update(self, batch: Batch):
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        mean_w = torch.as_tensor(self.blr.mu, dtype=torch.float32, device=self.device)
        phi = self.feature_net(obs)
        q_all = phi @ mean_w.t()
        q = q_all.gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_phi_online = self.feature_net(next_obs)
            next_q_online = next_phi_online @ mean_w.t()
            next_actions = next_q_online.argmax(dim=1)
            target_w = torch.as_tensor(self.blr.mu, dtype=torch.float32, device=self.device)
            next_phi_target = self.target_feature_net(next_obs)
            next_q_target = next_phi_target @ target_w.t()
            next_q = next_q_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
            y = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        loss = F.mse_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.feature_net.parameters(), 10.0)
        self.optim.step()
        return loss.item(), q.mean().item()

    @torch.no_grad()
    def update_posterior(self, batch: Batch) -> None:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        phi = self.feature_net(obs).cpu().numpy()
        next_phi = self.target_feature_net(next_obs).cpu().numpy()
        next_q = next_phi @ self.blr.mu.T
        targets = batch.rewards + self.cfg.gamma * (1.0 - batch.dones) * next_q.max(axis=1)
        self.blr.update(phi, batch.actions, targets)

    def save(self, path: str) -> None:
        torch.save({
            "cfg": self.cfg.__dict__,
            "feature_net": self.feature_net.state_dict(),
            "target_feature_net": self.target_feature_net.state_dict(),
            "optim": self.optim.state_dict(),
            "blr": self.blr.state_dict(),
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device)
        except Exception:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_net.load_state_dict(ckpt["feature_net"])
        self.target_feature_net.load_state_dict(ckpt["target_feature_net"])
        self.optim.load_state_dict(ckpt["optim"])
        self.blr.load_state_dict(ckpt["blr"])
        self.train_steps = int(ckpt.get("train_steps", 0))
