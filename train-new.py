import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import cv2 as cv
import json


def preprocess_image(img):
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    img = cv.resize(img, dsize=(84, 110), interpolation=cv.INTER_AREA)
    img = img[16:100]
    return img


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10 ** 6)
        self.training_frames = 10 ** 7
        self.image_sequence_size = 4
        self.frameskip = 4
        self.image_sequence = deque(maxlen=4)
        self.save_path = "save/"
        self.gamma = 0.95  # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 10 ** 6
        self.learning_rate = 0.00025
        try:
            self.load()
        except FileNotFoundError:
            self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, data_format='channels_last', activation='relu', input_shape=self.state_size))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, data_format='channels_last', activation='relu', input_shape=self.state_size))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='rmsprop')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, (-1, 84, 84, 4))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for i_state, action, reward, i_next_state, done in minibatch:
            i_state = np.reshape(i_state, (-1, 84, 84, 4))
            i_next_state = np.reshape(i_state, (-1, 84, 84, 4))
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(i_next_state)[0]))
            target_f = self.model.predict(i_state)
            target_f[0][action] = target
            self.model.fit(i_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self):
        with open(self.save_path + 'model_in_json.json', 'r') as f:
            model_json = json.load(f)

        self.model = model_from_json(model_json)
        self.model.load_weights(self.save_path + 'model_weights.h5')
        self.model.compile(loss='mse', optimizer='rmsprop')

    def save(self):
        model_json = self.model.to_json()
        with open(self.save_path + "model_in_json.json", "w") as json_file:
            json.dump(model_json, json_file)
        self.model.save_weights(self.save_path + "model_weights.h5")
        # self.model.save(path + "model.h5")


if __name__ == "__main__":
    env = gym.make('BreakoutNoFrameskip-v4')
    observation = preprocess_image(env.reset())
    state_size = (84, 84, 4)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    timesteps = 0
    i_episode = 0

    while True:
        if timesteps >= agent.training_frames:
            break
        score = 0
        observation = env.reset()
        agent.image_sequence.extend([observation, observation, observation, observation])
        for time in range(agent.training_frames):
            timesteps += 1
            # env.render()
            p_image = preprocess_image(observation)
            agent.image_sequence.append(p_image)

            action = 0

            if len(agent.image_sequence) < agent.image_sequence_size:
                next_observation, reward, done, _ = env.step(action)
            else:
                current_state = np.stack([agent.image_sequence[0],
                                          agent.image_sequence[1],
                                          agent.image_sequence[2],
                                          agent.image_sequence[3]])
                if time % 4 == 0:
                    action = agent.act(current_state)
                next_observation, reward, done, _ = env.step(action)
                score += reward
                reward = reward if not done else -10
                p_image = preprocess_image(next_observation)
                next_state = np.stack([agent.image_sequence[0],
                                       agent.image_sequence[1],
                                       agent.image_sequence[2],
                                       p_image])
                agent.remember(current_state, action, reward, next_state, done)
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            observation = next_observation
            if done:
                    print("episode: {}, total timesteps: {}, score: {}, e: {:.2}"
                        .format(i_episode, timesteps, score, agent.epsilon))
                    break

        if i_episode % 10 == 0:
            agent.save()
        i_episode += 1
    agent.save()