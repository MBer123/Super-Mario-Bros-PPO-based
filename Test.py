import os.path
from Super_Mario_Bros import SuperMarioEnv
import Basic
from CNN_PPO.Agent import PPO


if __name__ == "__main__":
    env_name = 'SuperMario'
    world = 1
    stage = 4

    env = SuperMarioEnv(render=True, train=True, world=world, stage=stage)

    # 用于保存模型的文件夹
    ckpt_dir = f"{world}-{stage}-finish"
    # ckpt_dir = f"SuperMario-PPO"

    agent = PPO(action_dim=env.action_space.n,
                 state_dim=env.observation_space.shape,
                 ckpt_dir=ckpt_dir)
    agent.load_models(230)

    Basic.test(env, agent, max_episodes=1000)
