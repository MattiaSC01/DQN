import numpy as np
import torch as th
import torch.nn as nn
from torchvision import transforms
from collections import namedtuple
import gymnasium as gym
import wandb


def epsilon_greedy(q_values: th.Tensor, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    return th.argmax(q_values).item()


class DQN(nn.Module):
    """
    A convolutional neural network that implements
    the state-action value function in a DQN agent.
    It takes as input a stack of num_frames frames,
    and outputs a value for each action.
    """

    def __init__(
        self,
        num_actions,
        num_frames=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # 20x20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 9x9
            nn.ReLU(),
            nn.Flatten(),  # 32 * 9 * 9 = 2592
        )
        self.head = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x

    def sample_action(self, frames, epsilon):
        with th.no_grad():
            q_values = self(frames.unsqueeze(0))
        return epsilon_greedy(q_values, epsilon)


KEYS = ["state", "action", "reward", "next_state", "done"]
SHAPES = [(1, 84, 84), (), (), (1, 84, 84), ()]
DTYPES = [th.float32, th.int32, th.float32, th.float32, th.bool]
Transition = namedtuple("Transition", KEYS)


class ReplayBuffer:
    """
    Data structure for storing and sampling
    transitions from the environment.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.keys = KEYS
        self.buffer = {
            key: th.zeros(
                (capacity, *shape),
                dtype=dtype,
            )
            for key, shape, dtype in zip(KEYS, SHAPES, DTYPES, strict=True)
        }
        self.idx = 0

    def __len__(self):
        return min(self.idx, self.capacity)

    def push(self, transition: Transition):
        for key in self.keys:
            self.buffer[key][self.idx % self.capacity] = getattr(transition, key)
        self.idx += 1

    # def sample(self, batch_size, num_frames):
    #     """
    #     Sample a batch of transitions from the buffer.
    #     Assumes that the buffer stores num_frames frames
    #     per time step in both `state` and `next_state`
    #     """
    #     indices = np.random.choice(len(self), batch_size, replace=False)
    #     batch = {key: self.buffer[key][indices] for key in self.keys}
    #     return Transition(**batch)

    def sample(self, batch_size, num_frames):
        """
        Sample a batch of transitions from the buffer.
        Assumes that the buffer stores only one frame
        per time step in both `state` and `next_state`
        """
        indices = np.random.choice(len(self) - num_frames, batch_size, replace=False)
        batch = {key: self.buffer[key][indices] for key in ["action", "reward", "done"]}
        history_indices = np.repeat(indices, num_frames).reshape(
            batch_size, num_frames
        ) + np.arange(num_frames)
        batch["state"] = self.buffer["state"][history_indices].squeeze(2)  # B, F, H, W
        batch["next_state"] = self.buffer["next_state"][history_indices].squeeze(2)
        return Transition(**batch)


class Logger:
    def __init__(self, project_name: str, config: dict):
        self.project_name = project_name
        self.config = config

    def begin(self):
        self.run = wandb.init(project=self.project_name, config=self.config)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log metrics to Weights and Biases.
        """
        wandb.log(metrics, step=step)

    def save_model(self, model: nn.Module, filename: str = "dqn.pth"):
        """
        Save the model checkpoint to Weights and Biases.
        """
        wandb.save(filename)
        th.save(model.state_dict(), filename)

    def finish(self):
        """
        End the Weights and Biases session
        """
        self.run.finish()


input_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((110, 84)),
        transforms.CenterCrop(84),
    ]
)


class Trainer:
    def __init__(
        self,
        model: DQN,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        num_frames: int = 4,
        device: str | th.device = "cpu",
        eval_interval: int = 500,
        eval_episodes: int = 25,
    ):
        self.model = model
        self.env = env
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.num_frames = num_frames
        self.optimizer = self.get_optimizer()
        self.epsilon = 1.0
        self.k = 4
        self.device = th.device(device)
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.model.to(self.device)

    def train(
        self,
        num_episodes: int = 10_000_000,
        learning_starts: int = 512,
        batch_size: int = 32,
        gamma: float = 1.0,
    ):
        self.episode = 0
        self.step = 0
        self.learning_starts = learning_starts
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size

        self.logger.begin()

        while self.episode < self.num_episodes:
            print(f"Episode {self.episode}")
            self.collect_rollout()
            if self.episode > self.learning_starts:
                train_loss = self.train_step()
                metrics = {"train/loss": train_loss, "epsilon": self.epsilon}
                if self.episode % self.eval_interval == 0:
                    cumulative_rewards, episode_lengths, rewards_per_step = (
                        self.evaluate()
                    )
                    metrics.update(
                        {
                            "eval/mean_reward": np.mean(cumulative_rewards),
                            "eval/std_reward": np.std(cumulative_rewards),
                            "eval/max_reward": np.max(cumulative_rewards),
                            "eval/mean_length": np.mean(episode_lengths),
                            "eval/std_length": np.std(episode_lengths),
                            "eval/max_length": np.max(episode_lengths),
                            "eval/mean_reward_per_step": np.mean(rewards_per_step),
                        }
                    )
                self.logger.log_metrics(
                    metrics,
                    step=self.episode,
                )
            self.episode += 1
            self.update_epsilon()

        self.end_training()

    def end_training(self):
        self.env.close()
        self.logger.save_model(self.model)
        self.logger.finish()

    def train_step(self):
        self.model.train()

        transitions = self.replay_buffer.sample(self.batch_size, self.num_frames)
        state = transitions.state.to(self.device)
        next_state = transitions.next_state.to(self.device)
        action = transitions.action.to(self.device)
        reward = self.scale_reward(transitions.reward).to(self.device)
        done = transitions.done.to(self.device)

        with th.no_grad():
            target = reward + self.gamma * th.max(self.model(next_state), dim=1).values
            target[done] = reward[done]  # terminal state

        q_values = self.model(state)
        pred = q_values[th.arange(self.batch_size), action]
        loss = th.mean((pred - target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @th.inference_mode()
    def evaluate(self):
        self.model.eval()
        cumulative_rewards, episode_lengths, rewards_per_step = [], [], []

        for _ in range(self.eval_episodes):
            observation, info = self.env.reset()
            observation = input_transform(observation)
            frames = th.cat(
                [th.zeros((self.num_frames - 1, 84, 84)), observation], dim=0
            )
            done = False
            cumulative_reward = 0
            episode_length = 0

            while not done:
                action = self.model.sample_action(frames.to(self.device), 0.05)
                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )
                next_observation = input_transform(next_observation)
                next_frames = th.cat([frames[1:], next_observation], dim=0)
                frames = next_frames
                observation = next_observation
                done = terminated or truncated
                cumulative_reward += reward
                episode_length += 1
            cumulative_rewards.append(cumulative_reward)
            episode_lengths.append(episode_length)
            rewards_per_step.append(cumulative_reward / episode_length)
        return cumulative_rewards, episode_lengths, rewards_per_step

    @th.no_grad()
    def collect_rollout(self):
        self.model.eval()
        observation, info = self.env.reset()
        observation = input_transform(observation)
        frames = th.cat([th.zeros((self.num_frames - 1, 84, 84)), observation], dim=0)
        done = False
        idx = 0

        while not done:
            if idx % self.k == 0:
                action = self.model.sample_action(frames.to(self.device), self.epsilon)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            next_observation = input_transform(next_observation)
            next_frames = th.cat([frames[1:], next_observation], dim=0)
            transition = Transition(
                state=frames[-1].clone(),
                action=action,
                reward=reward,
                next_state=next_frames[-1].clone(),
                done=terminated or truncated,
            )
            # transition = Transition(
            #     state=frames,
            #     action=action,
            #     reward=reward,
            #     next_state=next_frames,
            #     done=terminated or truncated,
            # )
            self.replay_buffer.push(transition)

            frames = next_frames
            observation = next_observation
            done = terminated or truncated
            self.step += 1
            idx += 1

    def make_checkpoint(self):
        th.save(self.model.state_dict(), "dqn.pth")

    def scale_reward(self, reward):
        return th.sign(reward)

    def get_optimizer(self):
        return th.optim.RMSprop(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

    def update_epsilon(self):
        self.epsilon = max(0.1, 1.0 - self.episode / self.num_episodes * 9)


if __name__ == "__main__":
    config = {
        "num_episodes": 100_000,
        "learning_starts": 256,
        "batch_size": 32,
        "gamma": 1.0,
        "game": "Pong-v4",
        "num_frames": 4,
        "device": "mps",
        "buffer_capacity": 10_000,
    }

    env = gym.make(config["game"])
    config["num_actions"] = env.action_space.n
    model = DQN(env.action_space.n, num_frames=config["num_frames"])
    replay_buffer = ReplayBuffer(config["buffer_capacity"])
    logger = Logger("DQN - Atari", config)
    trainer = Trainer(
        model,
        env,
        replay_buffer,
        logger,
        num_frames=config["num_frames"],
        device=config["device"],
    )
    trainer.train(
        num_episodes=config["num_episodes"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
    )
