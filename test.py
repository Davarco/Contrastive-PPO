import gym
import time

env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy', render=True)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        time.sleep(0.1)
        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
