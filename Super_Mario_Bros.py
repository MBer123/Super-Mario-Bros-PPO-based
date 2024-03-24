import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym import Env
from gym.spaces import Box
import numpy as np
import cv2


class SuperMarioEnv(Env):
    def __init__(self, render=False, train=True, max_step=1024, world=1, stage=1):
        super().__init__()
        self.world = world
        self.stage = stage
        self.game = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
        self.game = JoypadSpace(self.game, RIGHT_ONLY)

        self.RENDER = render
        self.TRAIN = train

        self.resized_height = 84
        self.resized_width = 84

        self.last_observation = None  # 用于存储前一帧的观察结果
        self.observation_space = Box(low=0, high=255, shape=(self.resized_width, self.resized_width, 2),
                                     dtype=np.uint8)
        self.action_space = self.game.action_space

        self.MAX_STEP = max_step

        # reward design
        self.pre_score = 0
        self.max_life = 2
        self.max_x_pos = 40
        self.steps = 0

    def reset(self):
        self.pre_score = 0
        self.steps = 0
        self.max_x_pos = 40
        self.max_life = 2
        self.last_observation = None

        return self.preprocess(self.game.reset())

    def step(self, action):
        """
            {'coins': 0, 'flag_get': False, 'life': 2,
            'score': 0, 'stage': 1, 'status': 'small',
            'time': 400, 'world': 1, 'x_pos': 40,
            'y_pos': 79, 'TimeLimit.truncated': False}
        """
        obs, reward, done, info = self.preprocess_action(action)

        # 杀敌和吃金币给予正反馈
        reward = 1 if info['score'] > self.pre_score else 0
        self.pre_score = info['score']

        # 向右移动给予正反馈
        if info['x_pos'] > self.max_x_pos:
            reward += info['x_pos'] - self.max_x_pos
            self.max_x_pos = info['x_pos']

        # 死亡重置环境
        self.max_life = max(self.max_life, info['life'])  # 因为角色可能获得 1up, 所以实时更新 life 值
        if info['life'] <= self.max_life - 1:
            done = True

        # 卡住重置环境
        self.steps += 1
        if self.steps >= self.MAX_STEP:
            done = True
            reward += 500

        # 游戏环境结束
        if done:
            # 赢得游戏给予正反馈, 失败不给予负反馈
            if info['flag_get']:
                reward += 500
            else:
                reward -= 500

            obs = self.reset()
            return obs, reward / 100, done, info

        return obs, reward / 100, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()

    def preprocess(self, observation):
        # 将图像转换为灰度
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        resize = cv2.resize(gray, (self.resized_height, self.resized_width), interpolation=cv2.INTER_CUBIC)

        # 归一化
        current_frame = resize / 255

        if self.last_observation is None:
            # 如果没有前一帧，就复制当前帧作为前一帧，确保输出始终有两帧
            self.last_observation = current_frame

        # 将当前帧和前一帧组合成一个新的观察结果
        combined_observation = np.stack((self.last_observation, current_frame), axis=-1)
        self.last_observation = current_frame  # 更新前一帧为当前帧

        return combined_observation

    def preprocess_action(self, action):
        # action == 0 -> 静止不动
        # action == 1 -> 往右走
        # action == 2 -> 往右小跳
        # action == 3 -> 往右跑
        # action == 4 -> 往右大跳
        # action == 5 -> 往上跳
        # action == 6 -> 往左走
        if action != 4 and action != 2:
            obs, _, done, info = self.game.step(action)

            # 处理各个关卡的局部最优解
            done = self.preprocess_stage_reward(info) or done

            if self.RENDER:
                self.render()
                if not self.TRAIN:
                    time.sleep(0.02)

        else:  # 重构跳跃
            obs, _, done, info = self.game.step(action)

            # 处理各个关卡的局部最优解
            done = self.preprocess_stage_reward(info) or done

            if self.RENDER:
                self.render()
                if not self.TRAIN:
                    time.sleep(0.02)

            # 记录执行跳跃动作后的y位置
            pre_y = info['y_pos']

            # 检查角色是否还在空中（即 y 位置是否发生变化）
            while True:
                if done:
                    break

                if action == 4:
                    obs, _, done, info = self.game.step(action)
                else:
                    obs, _, done, info = self.game.step(1)

                # 处理各个关卡的局部最优解
                done = self.preprocess_stage_reward(info) or done

                if self.RENDER:
                    self.render()
                    if not self.TRAIN:
                        time.sleep(0.02)

                # 如果 y 位置发生变化，更新 pre_y 并继续循环；
                # 否则，假设角色已经落地，跳出循环
                if pre_y == info['y_pos']:
                    break
                else:
                    pre_y = info['y_pos']

        obs = self.preprocess(obs)

        return obs, _, done, info

    def preprocess_stage_reward(self, info):
        done = False

        # 对于 world 1 state 2 关卡可能卡在局部最优解的特殊处理
        if self.world == 1 and self.stage == 2:
            if 900 <= info['x_pos'] <= 1000 and info['y_pos'] >= 130:
                done = True
            if 2650 <= info['x_pos'] <= 2700 and info['y_pos'] >= 160:
                done = True

        # 对于 world 1 state 3 关卡可能卡在局部最优解的特殊处理
        if self.world == 1 and self.stage == 3:
            if 640 <= info['x_pos'] <= 670 and info['y_pos'] <= 180:
                done = True

        # 对于 world 2 state 1 关卡可能卡在局部最优解的特殊处理
        if self.world == 2 and self.stage == 1:
            if 1950 <= info['x_pos'] <= 2100 and info['y_pos'] <= 85:
                done = True

        return done
