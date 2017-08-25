import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PS_IP = '127.0.0.1'
PS_PORT = 12222
TB_PORT = 12345
NUM_GLOBAL_STEPS = 90000000
SAVE_MODEL_SECS = 30
SAVE_SUMMARIES_SECS = 30
LOG_DIR = './log/'
NUM_WORKER = 4
VISUALISED_WORKERS = [0]  # e.g. [0] or [1,2]

_N_AVERAGE = 100

VSTR = 'V1.2'


class MyEnv:
    def __init__(self):
        self.spec = None
        self.metadata = {'semantics.autoreset': False}
        self.observation_space_shape = [2]
        self.action_space_n = 2

        self.max_step = 100
        self._ep_count = 0
        self._rewards_last_n_eps = deque(maxlen=_N_AVERAGE)

        self.p_me = 1
        self.p_he = 40
        self._state = [self.p_me, self.p_he]

    def reset(self):
        self._act_list = []
        self._ep_reward = 0
        return self._state

    def step(self, act):
        if act != 0 and act != 1:
            assert ValueError("act should only be 0 or 1")

        dur = self.p_he - self.p_me
        r = 1 if len(self._act_list) >= dur and self._act_list[-dur] == 1 else 0

        self._act_list.append(act)
        t = len(self._act_list) >= self.max_step
        i = {}

        self._ep_reward += r
        if t:
            self._ep_count += 1
            self._rewards_last_n_eps.append(self._ep_reward)
            i[VSTR+'/ep_count'] = self._ep_count
            i[VSTR+'/rew'] = self._ep_reward
            i[VSTR+'/aver_rew'] = np.average(self._rewards_last_n_eps)
            print(i)

        return self._state, r, t, i

    def render(self):
        t = ['o']
        for i in range(1, self._state[1] - self._state[0]):
            t.append(self._shoot(i))
        t.append('o')
        print(''.join(t))

    def _shoot(self, lastn):
        return '.' if len(self._act_list) >= lastn and self._act_list[-lastn] == 1 else ' '


def create_env(unused):
    env = MyEnv()
    return env


def main():
    env = create_env(0)
    t = True
    while True:
        if t:
            _, t = env.reset(), False
        _,_,t,_ = env.step(np.random.randint(2))
        env.render()

if __name__ == '__main__':
    main()
