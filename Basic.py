import time
import numpy as np
import os
import pandas as pd


# 定义一个函数用于创建目录。如果目录已存在，则打印已存在的消息；如果不存在，则创建该目录。
def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        sub_path = os.path.join(path, sub_dir)
        if os.path.exists(sub_path):
            print(sub_path + 'is already exist!')
        else:
            os.makedirs(sub_path, exist_ok=True)
            print(sub_path + 'create successfully!')


# 定义一个训练函数，用于训练强化学习中的智能体。
# 它接收环境（env），智能体（agent），检查点目录（ckpt_dir），最大时间步数，保存频率，以及环境数作为输入。
def train(env, agent, ckpt_dir, max_time_steps=5000000, save_frequency=5,
          each_episode_steps=512, num_envs=8):
    # 创建文件夹用于保存模型
    create_directory(ckpt_dir, sub_dirs=agent.main_net)

    total_rewards, avg_rewards = [], []
    total_reward = 0
    observation = env.reset()
    done_times = num_envs
    episode = 0

    for step in range(1, max_time_steps):
        state_value = agent.evaluate_state(observation)
        agent.state_value.extend(state_value)

        action = agent.choose_action(observation, isTrain=True)
        observation_, reward, done, infos = env.step(action)

        for d in done:
            if d:
                done_times += 1

        next_state_value = agent.evaluate_state(observation_)
        agent.next_state_value.extend(next_state_value)

        agent.remember(observation, action, reward, done)
        agent.learn()

        total_reward += sum(reward)
        observation = observation_

        if step % each_episode_steps == 0:
            total_reward = total_reward / done_times
            total_rewards.append(total_reward)
            avg_reward = np.mean(total_rewards[-100:])
            avg_rewards.append(avg_reward)
            print('EP:{} Reward:{} Avg_reward:{}'.
                  format(episode + 1, total_reward, avg_reward))
            total_reward = 0
            done_times = num_envs
            episode += 1

        if step % (each_episode_steps * save_frequency) == 0:
            agent.save_models(episode)
            df = pd.DataFrame(total_rewards, columns=['Total Rewards'])
            df.to_csv(os.path.join(ckpt_dir, "returns.csv"), index=False)

    df = pd.DataFrame(total_rewards, columns=['Total Rewards'])
    df.to_csv(os.path.join(ckpt_dir, "returns.csv"), index=False)


# 定义一个测试函数，用于测试智能体的性能。
# 它接收环境（env），智能体（agent），以及交互次数。
def test(env, agent, max_episodes=10):
    total_rewards, avg_rewards = [], []

    for episode in range(max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:

            action = agent.choose_action(observation, isTrain=False)
            observation_, reward, done, info = env.step(action)
            print(info)
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print('EP:{} Reward:{} Avg_reward:{} '.
              format(episode + 1, total_reward, avg_reward))

    env.close()
