from uav_bdqn_belief20.envs.search_track_env import SearchTrackBelief20Env, EnvConfig
from uav_bdqn_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig


def main():
    env = SearchTrackBelief20Env(EnvConfig(seed=0, macro_steps=3))
    obs, info = env.reset()
    print("obs", obs.shape, "info", {k: info[k] for k in ["detected", "completed", "known_targets", "visited_ratio"]})
    agent = BDQNAgent(BDQNConfig(obs_shape=env.observation_shape, action_dim=env.action_dim))
    for _ in range(80):
        a = agent.act(obs)
        next_obs, r, term, trunc, info = env.step(a)
        agent.replay.add(obs, a, r, next_obs, term or trunc)
        obs = next_obs
        if term or trunc:
            obs, info = env.reset()
    print("train_step", agent.train_step())
    print("final info", {k: info[k] for k in ["detected", "completed", "known_targets", "visited_ratio"]})
    print("Smoke test OK")


if __name__ == "__main__":
    main()
