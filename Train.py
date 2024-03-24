import os
import Basic
from CNN_PPO.Agent import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from Super_Mario_Bros import SuperMarioEnv


# 多进程训练
def make_env(rank, seed=0, render=False):
    def _init():
        env_ = SuperMarioEnv(render=render, world=2, stage=1)  # 您自定义的环境
        env_.seed(seed + rank)
        return env_

    return _init


if __name__ == "__main__":
    env_name = 'SuperMario'
    num_envs = 8  # num_envs >= 2
    env = SubprocVecEnv([make_env(i, render=True if i == 0 else False) for i in range(num_envs)])

    # 确保实验的可重复性和结果的一致性
    env.seed(42)

    # 用于保存模型的文件夹
    ckpt_dir = os.path.join(os.curdir, env_name + "-PPO")

    agent = PPO(action_dim=env.action_space.n,
                state_dim=env.observation_space.shape,
                ckpt_dir=ckpt_dir, K_epochs=10, gamma=0.9, ent_coef=0.02, learning_frequency=1024,
                epsilon_clip=0.2, lambda_=1.0)

    Basic.train(env, agent, ckpt_dir, max_time_steps=10000000, save_frequency=1,
                each_episode_steps=1024, num_envs=num_envs)
