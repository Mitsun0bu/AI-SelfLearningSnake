import torch
import random
import numpy as np
from collections import deque
from ai_controlled_game import SnakeGameAI, Direction, Point
from model import Linear_Qnet, QTrainer
from utils import plot

# Constants
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        # Randomness
        self.epsilon = 0
        # Discount rate, always has to be < 1
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game):
        ''' This function set the values of snake's states'''
        # Get the snake's head from the game
        head = game.snake[0]
        # Create 4 points 20px away from the head to detect danger
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        # Boolean to check what is the game direction
        # Only one of these variables is set to 1
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        # Create the state array
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food is left
            game.food.x > game.head.x,  # food is right
            game.food.y < game.head.y,  # food is up
            game.food.y > game.head.y,  # food is down
            ]
        # The use of dtype allows to convert True/False to 1/0
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        ''' This function keep the results of the game in memory '''
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        ''' This function train the AI for a batch of games '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        ''' This function train the AI for one game step only '''
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        ''' This function generate move
            either randomly or according to our prediciton model
        '''
        # Random moves : tradeoff between exploration and exploitation
        # Espilon randomness parameter will decrease while n_games increases
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        # Generate a random index i to modify final_move[i] randomly
        if random.randint(0, 200) < self.epsilon:
            i = random.randint(0, 2)
            final_move[i] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get old state
        state_old = agent.get_state(game)
        # Get move
        final_move = agent.get_action(state_old)
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            # Train the long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            # Save new highscore if we beat the previous one
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
