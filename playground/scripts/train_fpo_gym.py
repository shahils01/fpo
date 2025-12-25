import datetime
import time
from typing import Annotated

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as onp
import tyro
import wandb
from tqdm import tqdm

from flow_policy import fpo, rollouts


class GymEnvSpec:
    """Minimal spec shim to satisfy FpoState.init expectations."""

    def __init__(self, obs_size: int, action_size: int):
        self.observation_size = obs_size
        self.action_size = action_size


def make_vector_env(env_name: str, num_envs: int) -> gym.vector.VectorEnv:
    return gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])


def flatten_obs(obs: onp.ndarray) -> onp.ndarray:
    return obs.reshape(obs.shape[0], -1)


def gym_rollout(
    agent_state: fpo.FpoState,
    env: gym.vector.VectorEnv,
    iterations_per_env: int,
    episode_length: int,
    prng: jax.Array,
    deterministic: bool,
) -> rollouts.TransitionStruct:
    obs, _ = env.reset()
    obs = flatten_obs(onp.asarray(obs))
    obs_jnp = jnp.asarray(obs)

    traj = []
    steps = onp.zeros(env.num_envs, dtype=onp.int32)

    for _ in range(iterations_per_env):
        prng, prng_act = jax.random.split(prng)
        action, action_info, log_prob = agent_state.sample_action(
            obs_jnp, prng_act, deterministic=deterministic
        )
        # Env step expects numpy.
        action_env = onp.tanh(onp.asarray(action))
        next_obs, reward, terminated, truncated, _ = env.step(action_env)

        done_env = onp.asarray(terminated, dtype=bool)
        trunc_env = onp.asarray(truncated, dtype=bool)
        done_or_tr = onp.logical_or(done_env, trunc_env)

        steps = steps + 1
        discount = 1.0 - done_env.astype(onp.float32)  # keep 1 on truncation

        # Stack transition.
        traj.append(
            rollouts.TransitionStruct(
                obs=obs_jnp,
                next_obs=jnp.asarray(flatten_obs(next_obs)),
                action=action,
                action_info=action_info,
                log_prob=log_prob,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                truncation=jnp.asarray(done_or_tr, dtype=jnp.float32),
                discount=jnp.asarray(discount, dtype=jnp.float32),
            )
        )

        obs = flatten_obs(next_obs)
        obs[done_or_tr] = env.reset_done(done_or_tr)[0] if hasattr(env, "reset_done") else obs[done_or_tr]
        obs_jnp = jnp.asarray(obs)
        steps = onp.where(done_or_tr, 0, steps)

    transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *traj)
    return transitions


def main(
    env_name: str,
    wandb_entity: str,
    wandb_project: str,
    config: fpo.FpoConfig,
    exp_name: str = "",
    seed: int = 0,
    num_envs: int = 16,
    deterministic_eval: bool = True,
) -> None:
    """Train FPO on Gymnasium MuJoCo-style tasks."""
    vec_env = make_vector_env(env_name, num_envs=num_envs)
    obs_size = int(onp.prod(vec_env.single_observation_space.shape))
    action_size = int(onp.prod(vec_env.single_action_space.shape))

    env_spec = GymEnvSpec(obs_size, action_size)

    # Logging.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"fpo_gym_{env_name}_{exp_name}_{timestamp}",
        config={
            "env_name": env_name,
            "fpo_params": jdc.asdict(config),
            "learning_rate": config.learning_rate,
            "clipping_epsilon": config.clipping_epsilon,
            "seed": seed,
        },
    )

    agent_state = fpo.FpoState.init(
        prng=jax.random.key(seed),
        env=env_spec,
        config=config,
    )

    outer_iters = config.num_timesteps // (config.iterations_per_env * num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    prng = jax.random.key(seed + 1)
    times = [time.time()]
    for i in tqdm(range(outer_iters)):
        if i in eval_iters:
            transitions = gym_rollout(
                agent_state,
                vec_env,
                iterations_per_env=config.iterations_per_env,
                episode_length=config.episode_length,
                prng=prng,
                deterministic=deterministic_eval,
            )
            rewards = transitions.reward.sum(axis=0)
            wandb_run.log(
                {
                    "eval/reward_mean": onp.mean(onp.asarray(rewards)),
                    "eval/reward_std": onp.std(onp.asarray(rewards)),
                },
                step=i,
            )

        # Training rollout.
        transitions = gym_rollout(
            agent_state,
            vec_env,
            iterations_per_env=config.iterations_per_env,
            episode_length=config.episode_length,
            prng=prng,
            deterministic=False,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        wandb_run.log(
            {
                "train/mean_reward": onp.mean(onp.asarray(transitions.reward)),
                "train/mean_steps": (
                    transitions.discount.size
                    / jnp.sum(transitions.discount == 0.0)
                ),
                **{f"train/{k}": onp.mean(v) for k, v in metrics.items()},
            },
            step=i,
        )

        times.append(time.time())

    print("First train step time:", times[1] - times[0])
    print("~Train time:", times[-1] - times[1])


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.FlagConversionOff,))
