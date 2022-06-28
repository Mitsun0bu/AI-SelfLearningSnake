import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('metroid_vibes.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
GREEN1 = (0, 59, 0)
GREEN2 = (0, 143, 17)
GREEN3 = (0, 255, 65)
BLACK = (0, 0, 0)

# Constants
BLOCK_SIZE = 20
SPEED = 20


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        '''This method initializes the window attributes of SnakeGameAI'''
        # Init window width and height
        self.w = w
        self.h = h
        # Init screen for display and window caption
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnA.I.ke')
        # Create an object to help track time
        self.clock = pygame.time.Clock()
        # Reset game after game_over
        self.reset()

    def reset(self):
        '''This method resets SnakeGameAI after the end of a game'''
        # Init snake
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        # Init score, food position and i_frame
        self.score = 0
        self.food = None
        self._place_food()
        self.i_frame = 0

    def _place_food(self):
        '''This method generate random food position'''
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # If the food is on the snake, call the function again
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        '''This function enable snake action and update all variables'''
        # 0. Iterate game frame
        self.i_frame += 1
        # 1. Check if the user want to quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 2. Update the snake's head position
        # (insert method add updated snake's head at beginning of the snake list)
        self._move(action)
        self.snake.insert(0, self.head)
        # 3. Check if game over and update reward accordingly
        game_over = False
        reward = 0
        if self.is_collision() or self.i_frame > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # 4. If snake is eating :
        # -> Place new food and update score and reward accordingly
        # Else:
        # -> Just move
        # (pop method remove the last element of the snake list)
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        '''This method check for collision'''
        if pt is None:
            pt = self.head
        # Check if the snake hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check if the snake hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        '''This method update ui after each frame'''
        # Display background
        self.display.fill(BLACK)
        # Draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        # Draw the food
        pygame.draw.rect(self.display, GREEN3, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE - 5, BLOCK_SIZE - 5))
        # Display score
        text = font.render("Score: " + str(self.score), True, GREEN2)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        ''' This method make the snake move according to an action
            which is a list of booleans : [straight, right, left]
        '''
        # Define all the possible direction in clock_wise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # Find the current direction index in clock_wise list
        i = clock_wise.index(self.direction)
        # Find the new_direction index in clock_wise list
        if np.array_equal(action, [1, 0, 0]):  # no change
            new_direction = clock_wise[i]
        elif np.array_equal(action, [0, 1, 0]):  # elif right turn
            next_i = (i + 1) % 4
            new_direction = clock_wise[next_i]
        else:  # left turn
            next_i = (i - 1) % 4
            new_direction = clock_wise[next_i]
        # Update direction with new_direction
        self.direction = new_direction
        # Change the position of snake's heads according to direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
