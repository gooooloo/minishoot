import envs
import time

from baselines import deepq


def main():
    env = envs.create_env(None)
    act = deepq.load("{}_model.pkl".format(envs.VSTR))

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.1)
            obs, rew, done, _ = env.step(act([obs])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
