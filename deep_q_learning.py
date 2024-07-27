import random
import numpy as np
from collections import deque
import pygame
import tensorflow as tf
from game import clock, DOWN, LEFT, RIGHT, UP, Food, Snake, draw_objects, get_state


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation="relu")
        )
        model.add(tf.keras.layers.Dense(24, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model()


if __name__ == "__main__":
    agent = DQNAgent(state_size=11, action_size=4)
    episodes = 1000
    model_count = 0
    clock.tick(240)
    for e in range(episodes):
        try:
            snake = Snake()
            food = Food()
            state = get_state(snake, food)
            state = np.reshape(state, [1, 11])

            for time in range(60):
                action = agent.act(state)
                directions = [UP, DOWN, LEFT, RIGHT]
                snake.change_direction(directions[action])
                snake.move()
                reward = -0.1

                if snake.positions[0] == food.position:
                    reward = 1
                    snake.grow_snake()
                    food.spawn()

                if snake.check_collision():
                    reward = -1
                    done = True
                else:
                    done = False

                next_state = get_state(snake, food)
                next_state = np.reshape(next_state, [1, 11])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                draw_objects(snake, food)

                if done:
                    print(f"Episode: {e}/{episodes}, Score: {len(snake.positions)}")
                    break

                model_count += 1
                agent.replay()
        except (KeyboardInterrupt, SystemExit, Exception, SystemError):
            break
    agent.model.save("snake_model.keras")
    pygame.quit()
