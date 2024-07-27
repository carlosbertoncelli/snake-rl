import numpy as np
import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 640, 480
cell_size = 20

# Colors
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
black = (0, 0, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Screen setup
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 35)


class Snake:
    def __init__(self):
        self.positions = [(100, 100), (80, 100), (60, 100)]
        self.direction = RIGHT
        self.grow = False

    def move(self):
        if self.grow:
            self.grow = False
        else:
            self.positions.pop()

        head_x, head_y = self.positions[0]
        new_head = (
            head_x + self.direction[0] * cell_size,
            head_y + self.direction[1] * cell_size,
        )
        self.positions.insert(0, new_head)

    def change_direction(self, direction):
        if direction == UP and self.direction != DOWN:
            self.direction = UP
        elif direction == DOWN and self.direction != UP:
            self.direction = DOWN
        elif direction == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif direction == RIGHT and self.direction != LEFT:
            self.direction = RIGHT

    def grow_snake(self):
        self.grow = True

    def check_collision(self):
        head = self.positions[0]
        return (
            head[0] in (0, width)
            or head[1] in (0, height)
            or head in self.positions[1:]
        )


class Food:
    def __init__(self):
        self.position = (
            random.randint(0, (width - cell_size) // cell_size) * cell_size,
            random.randint(0, (height - cell_size) // cell_size) * cell_size,
        )

    def spawn(self):
        self.position = (
            random.randint(0, (width - cell_size) // cell_size) * cell_size,
            random.randint(0, (height - cell_size) // cell_size) * cell_size,
        )


def draw_objects(snake, food):
    screen.fill(black)
    for pos in snake.positions:
        pygame.draw.rect(
            screen, green, pygame.Rect(pos[0], pos[1], cell_size, cell_size)
        )
    pygame.draw.rect(
        screen,
        red,
        pygame.Rect(food.position[0], food.position[1], cell_size, cell_size),
    )
    pygame.display.flip()


def get_state(snake, food):
    head_x, head_y = snake.positions[0]
    food_x, food_y = food.position
    state = [
        head_x,
        head_y,
        food_x,
        food_y,
        snake.direction[0],
        snake.direction[1],
        int(snake.direction == UP),
        int(snake.direction == DOWN),
        int(snake.direction == LEFT),
        int(snake.direction == RIGHT),
        len(snake.positions),
    ]
    return np.array(state)


def main():
    snake = Snake()
    food = Food()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.change_direction(UP)
                elif event.key == pygame.K_DOWN:
                    snake.change_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    snake.change_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction(RIGHT)

        snake.move()

        if snake.positions[0] == food.position:
            snake.grow_snake()
            food.spawn()

        if snake.check_collision():
            running = False

        draw_objects(snake, food)
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
