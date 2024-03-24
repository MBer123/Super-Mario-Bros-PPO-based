import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.distributions import Categorical, Normal
from torch.utils.data import TensorDataset, DataLoader
import datetime


# 确保实验的可重复性和结果的一致性
def set_seed(seed: int = 42) -> None:
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
set_seed()


class CNN(nn.Module):
    """
    :param lr: 学习率
    :param state_dim: 状态空间维度
    """
    def __init__(self, lr, state_dim):
        super().__init__()

        in_channels = state_dim[2]
        self.state_dim = state_dim
        self.cnn_layers = [nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()]
        self.cnn_layers += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()]
        self.cnn_layers += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()]
        self.cnn_layers += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()]

        self.cnn_layers = nn.Sequential(*self.cnn_layers).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def calculate_conv_output_dims(self):
        state = T.zeros(1, self.state_dim[2], self.state_dim[0], self.state_dim[1]).to(device)
        dims = self.cnn_layers(state)

        return int(np.prod(dims.size()))

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        state = self.cnn_layers(state)

        extracted_feature = state.reshape(state.size(0), -1)

        return extracted_feature

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class ActorNetwork(nn.Module):
    """
    :param lr: 学习率
    :param extracted_feature_dim: 状态空间维度
    :param action_dim: 动作空间维度
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    :param is_continuous: 是否为连续动作空间
    """
    def __init__(self, lr, extracted_feature_dim, action_dim, fc1_dim, is_continuous):
        super(ActorNetwork, self).__init__()

        self.is_continuous = is_continuous

        self.layers = [nn.Linear(extracted_feature_dim, fc1_dim)]

        self.layers = nn.Sequential(*self.layers)

        if is_continuous:
            self.action_mean = nn.Linear(fc1_dim, action_dim)
            self.action_log_std = nn.Linear(fc1_dim, action_dim)
        else:
            self.action_prob = nn.Linear(fc1_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, extracted_feature):
        x = self.layers(extracted_feature)

        if self.is_continuous:
            action_mean = F.tanh(self.action_mean(x))
            action_log_std = self.action_log_std(x)
            action_log_std = T.clamp(action_log_std, float('-inf'), 1)

            return action_mean, action_log_std
        else:
            return F.softmax(self.action_prob(x), dim=-1)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    """
    :param lr: 学习率
    :param state_dim: 状态空间维度
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    """
    def __init__(self, lr, extracted_feature_dim, fc1_dim):
        super(CriticNetwork, self).__init__()

        self.layers = [nn.Linear(extracted_feature_dim, fc1_dim)]
        self.layers = nn.Sequential(*self.layers)

        self.V = nn.Linear(fc1_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, extracted_feature):
        x = self.layers(extracted_feature)

        return self.V(x)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class PPO:
    """
    :param state_dim: 环境状态空间的维度
    :param action_dim: 环境动作空间的维度
    :param ckpt_dir: 保存模型权重文件的路径
    :param gamma: 折扣因子
    :param is_continuous: 指示动作空间是连续的还是离散的
    :param actor_lr: Actor网络的学习率
    :param critic_lr: Critic网络的学习率
    :param actor_fc1_dim: Actor网络第一个全连接层的维度
    :param actor_fc2_dim: Actor网络第二个全连接层的维度
    :param critic_fc1_dim: Critic网络第一个全连接层的维度
    :param critic_fc2_dim: Critic网络第二个全连接层的维度
    :param K_epochs: 每次学习过程中的epochs数量
    :param epsilon_clip: 裁剪参数,控制新旧策略偏离程度
    :param learning_frequency: 多少步后进行一次学习
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, actor_lr=0.0001, critic_lr=0.0001,
                 actor_fc1_dim=512, critic_fc1_dim=512, ent_coef=0.01, gamma=0.99,
                 K_epochs=20, epsilon_clip=0.1, is_continuous=False, learning_frequency=1024,
                 lambda_=0.95):
        self.checkpoint_dir = ckpt_dir

        self.main_net = ['Actor-net', 'Critic-net', 'CNN-net']
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.state_value = []
        self.next_state_value = []

        self.log_prob_memory = []
        self.terminate_memory = []

        self.count = 0
        self.is_continuous = is_continuous
        self.learning_frequency = learning_frequency

        # 初始化Actor网络和Critic网络
        self.cnn = CNN(lr=critic_lr, state_dim=state_dim)
        extracted_feature_dim = self.cnn.calculate_conv_output_dims()

        self.actor = ActorNetwork(lr=actor_lr, extracted_feature_dim=extracted_feature_dim,
                                  action_dim=action_dim, fc1_dim=actor_fc1_dim, is_continuous=is_continuous)
        self.critic = CriticNetwork(lr=critic_lr, extracted_feature_dim=extracted_feature_dim,
                                    fc1_dim=critic_fc1_dim)

    def remember(self, state, action, reward, done):
        self.state_memory.extend(state)
        self.action_memory.extend(action)
        self.reward_memory.extend(reward)
        self.terminate_memory.extend(done)
        self.count += 1

    def choose_action(self, observation, isTrain=True):
        if isTrain:
            state = T.tensor(observation, dtype=T.float).to(device)
        else:
            state = T.tensor([observation], dtype=T.float).to(device)

        extracted_feature = self.cnn(state)

        if self.is_continuous:
            action_mean, action_log_prob = self.actor(extracted_feature)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)

            dist = Normal(action_mean, action_std)
            action = dist.sample()  # 这会为每个动作维度独立采样

            log_prob = dist.log_prob(action).sum(dim=-1)

            if isTrain:
                self.log_prob_memory.extend(log_prob)

            action = np.ravel(action.detach().cpu())
        else:
            probabilities = self.actor(extracted_feature).squeeze()
            dist = Categorical(probabilities)
            action = dist.sample()
            log_prob = dist.log_prob(action).tolist()

            if isTrain:
                self.log_prob_memory.extend(log_prob)

            action = action.tolist()

        return action

    def evaluate_action(self, state_tensor, action_tensor):
        extracted_feature = self.cnn(state_tensor)
        if self.is_continuous:
            action_mean, action_log_prob = self.actor(extracted_feature)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)
            dist = Normal(action_mean, action_std)

            dist_entropy = dist.entropy().sum(dim=-1)
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        else:
            probabilities = self.actor(extracted_feature).squeeze()
            dist = Categorical(probabilities)

            dist_entropy = dist.entropy()
            log_prob = dist.log_prob(action_tensor)

        return log_prob, dist_entropy

    def evaluate_state(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).to(device)
            extracted_feature = self.cnn(state)

            state_value = self.critic(extracted_feature).squeeze()

        return state_value.tolist()

    def learn(self):
        if self.count < self.learning_frequency:
            return

        # 转换为张量
        state_array = np.array(self.state_memory)
        state_tensor = T.tensor(state_array, dtype=T.float).to(device)

        action_tensor = T.tensor(self.action_memory, dtype=T.float).to(device)
        reward_tensor = T.tensor(self.reward_memory, dtype=T.float).to(device)

        with T.no_grad():
            log_prob_tensor = T.tensor(self.log_prob_memory, dtype=T.float).squeeze().to(device)
            state_value = T.tensor(self.state_value, dtype=T.float).squeeze().to(device)
            next_state_value = T.tensor(self.next_state_value, dtype=T.float).squeeze().to(device)
            terminate_tensor = T.tensor(self.terminate_memory, dtype=T.float).squeeze().to(device)

            delta = reward_tensor + self.gamma * next_state_value * (1 - terminate_tensor) - state_value

            gae = 0
            advantage = []
            gamma_lambda = self.gamma * self.lambda_
            for delta in reversed(delta.detach().cpu().numpy()):
                gae = delta + gamma_lambda * gae
                advantage.insert(0, gae)

            advantage = T.tensor(advantage, dtype=T.float32).to(device)
            R = advantage + state_value

        dataset = TensorDataset(state_tensor, action_tensor, advantage, R, log_prob_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for _ in range(self.K_epochs):
            for state_tensor, action_tensor,  advantage, R, log_prob_tensor in dataloader:
                # 评估新的对数概率和熵
                new_log_prob_tensor, dist_entropy = self.evaluate_action(state_tensor, action_tensor)
                new_log_prob_tensor = new_log_prob_tensor.squeeze()
                dist_entropy = dist_entropy.squeeze()

                # 计算新的状态值
                extracted_feature = self.cnn(state_tensor)
                new_state_value = self.critic(extracted_feature).squeeze()

                # 计算重要性采样比率
                ratios = T.exp(new_log_prob_tensor - log_prob_tensor)

                # 计算代理损失和值损失
                surrogate_1 = ratios * advantage
                surrogate_2 = T.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage

                actor_loss = -T.min(surrogate_1, surrogate_2)
                critic_loss = F.mse_loss(new_state_value, R)

                # 总损失 = actor 损失 + critic 损失 - 熵损失
                loss = T.mean(actor_loss + 0.5 * critic_loss - self.ent_coef * dist_entropy)

                self.critic.optimizer.zero_grad()
                self.actor.optimizer.zero_grad()
                self.cnn.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.cnn.parameters(), 0.5)
                self.critic.optimizer.step()
                self.actor.optimizer.step()
                self.cnn.optimizer.step()

        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

        self.state_value.clear()
        self.next_state_value.clear()

        self.log_prob_memory.clear()
        self.terminate_memory.clear()
        self.count = 0

    def save_models(self, episode):
        self.actor.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Saving actor network successfully!')
        self.critic.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                 "critic_{}.pt").format(episode))
        print('Saving critic network successfully!')
        self.cnn.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[2],
                                                 "cnn_{}.pt").format(episode))
        print('Saving cnn network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Loading actor network successfully!')
        self.critic.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                 "critic_{}.pt").format(episode))
        print('Loading critic network successfully!')
        self.cnn.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[2],
                                                 "cnn_{}.pt").format(episode))
        print('Loading cnn network successfully!')
