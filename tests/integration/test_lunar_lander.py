import gymnasium as gym
import numpy as np

from nova.nn import NN, Activation


class TestLunarLander:

    def test_run(self):
        test_env = "LunarLander-v2"
        random_seed = 42
        epochs = 5
        max_batch = 5
        env = gym.make(test_env)
        actions = [
            action for action in range(env.action_space.start, env.action_space.n)
        ]

        observation_dim, action_dim = env.observation_space.shape[0], env.action_space.n
        policy = NN(
            arch=[observation_dim, 4, 6, action_dim],
            activations=[Activation.tanh, Activation.tanh, Activation.identity],
        )

        for epoch in range(epochs):
            obs, _ = env.reset(seed=random_seed)
            current_batch = 0
            while True:
                # Execute trajectories and collect data for training
                probs = np.exp(policy.forward(obs))
                probs /= probs.sum()

                action = np.random.choice(actions, p=probs)
                obs, rew, term, trunc, _ = env.step(action)

                if term or trunc:
                    obs, _ = env.reset()
                    current_batch += 1

                    if current_batch >= max_batch:
                        break

            # Perform training
