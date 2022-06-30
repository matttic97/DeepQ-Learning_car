from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from collections import deque
import tensorflow as tf
import numpy as np
import random
import pickle as pickle


class DQNAgent:
    def __init__(self, gamma, n_actions, epsilon, batch_size, input_dims, min_mem_size,
                 max_mem_size=25000, update_target_every = 10, epsilon_dec=0.9995, epsilon_min=0.1,
                 agent_path=None, replay_memory_path=None):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.min_replay_memory_size = min_mem_size
        self.input_dims = input_dims

        # main model - for training
        if agent_path:
            self.model = keras.models.load_model(agent_path)
        else:
            self.model = self.create_model(input_dims, n_actions)

        # target model - for predicting
        self.target_model = self.create_model(input_dims, n_actions)

        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.target_update_every = update_target_every

        if replay_memory_path:
            with open(replay_memory_path, 'rb') as file:
                self.replay_memory = pickle.load(file)
        else:
            self.replay_memory = deque(maxlen=max_mem_size)

    def create_model(self, observation_space, n_action):
        model = Sequential()
        model.add(keras.Input(shape=observation_space))
        model.add(Dense(256, activation=tf.nn.relu))
        model.add(Dense(n_action, activation=tf.nn.softmax))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.005), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def choose_action(self, state):
        state = np.array(state)
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([transition[0] for transition in batch])
        #current_states = current_states.reshape(-1, current_states.shape[1], current_states.shape[2], 1)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        #new_current_states = new_current_states.reshape(-1, new_current_states.shape[1], new_current_states.shape[2], 1)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    # Queries main network for Q values given current observation space (environment state)
    def get_max_q(self, state):
        return np.argmax(self.model.predict(tf.reshape(state, [1, self.input_dims])).flatten())
