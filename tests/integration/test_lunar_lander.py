import gymnasium as gym
import matplotlib.pyplot as plt
from nova.nn import NN, Activation


class TestLunarLander:
    TEST_ENV = "LunarLander-v2"

    def test_run(self):
        env = gym.make("LunarLander-v2")
        trajs = []
        for _ in range(10):
            traj = []
            term, trunc = False, False
            obs, _ = env.reset(seed=42)
            while not term or not trunc:
                a = env.action_space.sample()
                _obs, rew, term, trunc, _ = env.step(a)

                traj.append((obs, a, rew))
                obs = _obs
            trajs.append(traj)

        for traj in trajs:
            x = []
            y = []
            rew = 0.0
            for step in traj:
                obs, _, r = step
                x.append(obs[0])
                y.append(obs[1])
                rew += r

            plt.plot(x, y, label=f"{r:.2f}")
        plt.legend()
        plt.show()
