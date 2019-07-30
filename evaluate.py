from train import *

EPISODES = 1
MAX_FRAMES = 10 ** 7

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    state = preprocess_image(env.reset())
    state_size = [84, 84]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0
    done = False
    batch_size = 32
    timesteps = 0

    for i_episode in range(EPISODES):
        score = 0
        state = preprocess_image(env.reset())
        state = np.reshape(state, [1, 1] + state_size)
        action = None
        for time in range(MAX_FRAMES):
            timesteps += 1
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done else -10
            next_state = preprocess_image(next_state)
            next_state = np.reshape(next_state, [1, 1] + state_size)
            state = next_state
            if done:
                print("episode: {}, total timesteps: {}, score: {}"
                      .format(i_episode, timesteps, score))
                break
        i_episode += 1