import envs
import models
import simple


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 60.5
    return is_solved


def main():
    env = envs.create_env(None)
    model = models.mlp([64])
    act = simple.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.01,
        exploration_final_eps=0.0,
        print_freq=10,
        callback=callback
    )
    print("Saving model to {}_model.pkl".format(envs.VSTR))
    act.save("{}_model.pkl".format(envs.VSTR))


if __name__ == '__main__':
    main()
