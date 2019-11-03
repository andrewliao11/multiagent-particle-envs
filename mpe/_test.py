import mpe
import time
import numpy as np


def savegif(arrs):
    import imageio
    imageio.mimsave('test.gif', arrs)


def agent_fn(entity):
    entity.initial_mass = 100.0
    

kwargs = {
    'seed': 123, 
    'agent_config': {'change_fn': agent_fn}
}
env = mpe.make_env("simple_spread", **kwargs)

rgb_arrs = []
for i in range(100):
    act = np.zeros(5)
    act[3] = 1
    env.step([act for i in range(3)])
    rgb_arr = env.render(mode='rgb_array')
    rgb_arrs.append(rgb_arr[0])


savegif(rgb_arrs)

