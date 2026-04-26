import sys

import pygame
import random
import math
import numpy as np


class SnakeEnv:
    def __init__(self, grid_width=20, grid_height=20, render=False):
        """初始化贪吃蛇环境
        参数:
            grid_width, grid_height: 游戏网格尺寸
            render: 是否渲染游戏界面
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.render_enabled = render

        # 常量
        self.ACTIONS = {
            0: (0, -1),  # 上
            1: (0, 1),  # 下
            2: (-1, 0),  # 左
            3: (1, 0)  # 右
        }
        self.n_actions = len(self.ACTIONS)

        # 游戏状态
        self.reset()

        # 渲染相关
        if render:
            self.init_render()

    def init_render(self):
        """初始化渲染组件"""
        pygame.init()
        self.GRID_SIZE = 20
        self.WIDTH = self.grid_width * self.GRID_SIZE
        self.HEIGHT = self.grid_height * self.GRID_SIZE
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('DQN 贪吃蛇')
        self.CLOCK = pygame.time.Clock()
        self.FONT = pygame.font.Font(None, 24)

    def reset(self):
        """重置环境，返回初始状态"""
        # 蛇的初始位置和方向
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = (1, 0)  # 向右
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_width * self.grid_height * 2  # 防止无限循环
        self.food = self._generate_food()
        self.done = False
        self.last_distance = self._get_distance_to_food()

        return self._get_state_mlp()

    def _generate_food(self):
        """生成不在蛇身上的食物"""
        while True:
            food = (random.randint(0, self.grid_width - 1),
                    random.randint(0, self.grid_height - 1))
            if food not in self.snake:
                return food

    def _get_distance_to_food(self):
        """计算蛇头到食物的欧氏距离"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2)

    def _get_state(self):
        """获取当前游戏状态，作为神经网络的输入"""
        # 方法1: 网格表示法 (推荐用于CNN)
        state = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.float32)

        # 食物位置 (红色通道)
        food_x, food_y = self.food
        state[food_y, food_x, 0] = 1.0

        # 蛇身位置 (绿色通道)
        for x, y in self.snake[1:]:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                state[y, x, 1] = 1.0

        # 蛇头位置 (蓝色通道)
        head_x, head_y = self.snake[0]
        if 0 <= head_x < self.grid_width and 0 <= head_y < self.grid_height:
            state[head_y, head_x, 2] = 1.0

        return state

    def _get_state_mlp(self):
        """
        返回一个 12 维的特征向量，适合 MLP 输入
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        has_body_up = any(seg[0] == head_x and seg[1] < head_y for seg in self.snake[1:])
        has_body_down = any(seg[0] == head_x and seg[1] > head_y for seg in self.snake[1:])
        has_body_left = any(seg[1] == head_y and seg[0] < head_x for seg in self.snake[1:])
        has_body_right = any(seg[1] == head_y and seg[0] > head_x for seg in self.snake[1:])

        # 1. 食物相对方向（归一化）
        dx = (food_x - head_x) / self.grid_width
        dy = (food_y - head_y) / self.grid_height

        # 2. 四个方向是否安全（0=危险，1=安全）
        danger_straight = 0
        danger_left = 0
        danger_right = 0

        # 当前方向映射
        if self.direction == (0, -1):  # 上
            front = (head_x, head_y - 1)
            left = (head_x - 1, head_y)
            right = (head_x + 1, head_y)
        elif self.direction == (0, 1):  # 下
            front = (head_x, head_y + 1)
            left = (head_x + 1, head_y)
            right = (head_x - 1, head_y)
        elif self.direction == (-1, 0):  # 左
            front = (head_x - 1, head_y)
            left = (head_x, head_y + 1)
            right = (head_x, head_y - 1)
        elif self.direction == (1, 0):  # 右
            front = (head_x + 1, head_y)
            left = (head_x, head_y - 1)
            right = (head_x, head_y + 1)

        # 检查前方是否危险
        if (front[0] < 0 or front[0] >= self.grid_width or
                front[1] < 0 or front[1] >= self.grid_height or
                front in self.snake[1:]):
            danger_straight = 1

        # 检查左方是否危险
        if (left[0] < 0 or left[0] >= self.grid_width or
                left[1] < 0 or left[1] >= self.grid_height or
                left in self.snake[1:]):
            danger_left = 1

        # 检查右方是否危险
        if (right[0] < 0 or right[0] >= self.grid_width or
                right[1] < 0 or right[1] >= self.grid_height or
                right in self.snake[1:]):
            danger_right = 1

        # 3. 当前运动方向（one-hot）
        dir_up = 1 if self.direction == (0, -1) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0
        dir_right = 1 if self.direction == (1, 0) else 0

        # 4. 蛇长度（归一化）
        snake_length = len(self.snake) / (self.grid_width * self.grid_height)

        # 5. 是否靠近边界（可选，增强泛化）
        near_wall = (
            int(head_x == 0 or head_x == self.grid_width - 1 or
                head_y == 0 or head_y == self.grid_height - 1)
        )

        state = np.array([
            dx,  # 食物x相对位置
            dy,  # 食物y相对位置
            danger_straight,  # 正前方是否危险
            danger_left,  # 左边是否危险
            danger_right,  # 右边是否危险
            dir_up,  # 朝上
            dir_down,  # 朝下
            dir_left,  # 朝左
            dir_right,  # 朝右
            snake_length,  # 蛇长度
            near_wall,  # 是否在边界
            has_body_up,
            has_body_left,
            has_body_right,
            has_body_down,
            self.steps / self.max_steps  # 时间压力（可选）
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        执行动作，返回 (next_state, reward, done, info)
        """
        self.steps += 1
        self.last_distance = self._get_distance_to_food()
        # 1. 根据动作更新方向 (防止180度转向)
        new_direction = self.ACTIONS[action]
        # 检查是否是180度转向
        if (new_direction[0] + self.direction[0] == 0 and
                new_direction[1] + self.direction[1] == 0):
            # 无效转向，保持当前方向
            new_direction = self.direction

        self.direction = new_direction

        # 2. 计算新头部位置
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # 3. 检查游戏是否结束
        reward = 0
        done = False
        info = {}

        reward += 0.05 if self.score > 0 else 0
        # 检查是否撞墙
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or
                new_head[1] < 0 or new_head[1] >= self.grid_height):
            reward -= 10  # 撞墙惩罚
            done = True
            info['termination'] = 'wall_collision'

        # 检查是否撞到自己
        elif new_head in self.snake[1:]:
            reward -= 10  # 撞到自己惩罚
            done = True
            info['termination'] = 'self_collision'

        # 检查是否超时
        elif self.steps > self.max_steps:
            reward -= 10  # 超时惩罚
            done = True
            info['termination'] = 'timeout'

        # 4. 正常移动
        else:
            self.snake.insert(0, new_head)

            # 检查是否吃到食物
            if new_head == self.food:
                reward += 10  # 吃到食物奖励
                self.score += 10
                self.food = self._generate_food()
                # 重置最大步数
                self.max_steps = max(self.max_steps, self.steps + self.grid_width * self.grid_height)
                self.last_distance = self._get_distance_to_food()
            else:
                # 没有吃到食物，移除尾部
                self.snake.pop()
                # new_distance = self._get_distance_to_food()
                # if new_distance < self.last_distance:
                #     reward = +1
                # else:
                #     reward = -1
                #
                # self.last_distance = new_distance
            # 5. 小步惩罚，鼓励高效行动
            reward += 0.01

        # 6. 获取下一个状态
        next_state = self._get_state_mlp()
        self.done = done

        # 7. 渲染
        if self.render_enabled:
            self.render()

        return next_state, reward, done, info

    def render(self, mode='human'):
        """渲染游戏界面"""
        if not self.render_enabled:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 绘制背景
        self.SCREEN.fill((0, 0, 0))

        # 绘制网格
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.SCREEN, (40, 40, 40), (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.SCREEN, (40, 40, 40), (0, y), (self.WIDTH, y))

        # 绘制食物
        food_x, food_y = self.food
        pygame.draw.rect(self.SCREEN, (255, 0, 0),
                         (food_x * self.GRID_SIZE, food_y * self.GRID_SIZE,
                          self.GRID_SIZE, self.GRID_SIZE))

        # 绘制蛇
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.SCREEN, color,
                             (x * self.GRID_SIZE, y * self.GRID_SIZE,
                              self.GRID_SIZE, self.GRID_SIZE))

        # 显示分数和步数
        score_text = self.FONT.render(f'分数: {self.score}', True, (255, 255, 255))
        steps_text = self.FONT.render(f'步数: {self.steps}', True, (255, 255, 255))
        self.SCREEN.blit(score_text, (10, 10))
        self.SCREEN.blit(steps_text, (10, 30))

        pygame.display.flip()
        self.CLOCK.tick(120)  # 控制渲染速度