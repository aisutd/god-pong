import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import cv2 as cv
import json

EPISODES = 1000
MAX_FRAMES = 10 ** 7
IMAGE_SEQUENCE_SIZE = 4
SCORES = []
METRIC_STATES = []
path = "save/"


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
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 10 ** 6
        self.learning_rate = 0.00025
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, data_format='channels_first', activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, data_format='channels_first', activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='rmsprop')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self):
        with open(path + 'model_in_json.json', 'r') as f:
            model_json = json.load(f)

        self.model = model_from_json(model_json)
        self.model.load_weights(path + 'model_weights.h5')
        self.model.compile(loss='mse', optimizer='rmsprop')

    def save(self):
        model_json = self.model.to_json()
        with open(path + "model_in_json.json", "w") as json_file:
            json.dump(model_json, json_file)
        self.model.save_weights(path + "model_weights.h5")
        # self.model.save(path + "model.h5")


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    state = preprocess_image(env.reset())
    state_size = [84, 84]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    del agent.model
    agent.load()

    done = False
    batch_size = 32
    timesteps = 0
    image_sequence = []

    for e in range(EPISODES):
        if timesteps >= MAX_FRAMES:
            break
        score = 0
        state = preprocess_image(env.reset())
        state = np.reshape(state, [1, 1] + state_size)
        action = None
        for time in range(MAX_FRAMES):
            timesteps += 1
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done else -10
            next_state = preprocess_image(next_state)
            image_sequence.append(next_state)
            if len(image_sequence) > IMAGE_SEQUENCE_SIZE:
                image_sequence.pop(0)
                current_state = np.stack([image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3]])
            next_state = np.reshape(next_state, [1, 1] + state_size)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}, total timesteps: {}, score: {}, e: {:.2}"
                      .format(e, timesteps, score, agent.epsilon))
                SCORES.append(score)

                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save()
    agent.save()
