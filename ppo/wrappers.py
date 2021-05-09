from gym.envs.classic_control import rendering
import numpy as np
import gym


class BaseWrapper():
    def __init__(self, venv):
        self.venv = venv
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.viewer = rendering.SimpleImageViewer()

    def reset(self):
        pass
    
    def step_wait(self):
        pass

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        return self.venv.close()

    def render(self):
        render = self.venv.render('rgb_array')
        self.viewer.imshow(render)


class ProcgenWrapper(BaseWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = venv.observation_space['rgb']
        obs_shape = self.observation_space.shape
        self.observation_space.shape = (obs_shape[2], obs_shape[0], obs_shape[1])

    def reset(self):
        obs = self.venv.reset()['rgb']
        obs = obs.transpose(0, 3, 1, 2)
        obs = obs/255
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        obs = obs['rgb']
        obs = obs.transpose(0, 3, 1, 2)
        obs = obs/255
        # rewards = np.sign(rewards)

        return obs, rewards, dones, infos
    
