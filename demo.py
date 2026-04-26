import pygame
import torch
import numpy as np
import os
import time
from snake_env import SnakeEnv
from dqn_agent import DQNAgent


def demo_trained_agent(model_path, episodes=10):
    """使用训练好的模型进行演示"""
    print(f"尝试加载模型: {model_path}")

    # 创建渲染环境
    env = SnakeEnv(grid_width=20, grid_height=20, render=True)

    # 确保渲染已初始化
    if not hasattr(env, 'SCREEN') and env.render_enabled:
        env.init_render()

    # 创建智能体
    agent = DQNAgent(env)

    # 加载训练好的模型，如果不存在则使用随机策略
    if os.path.exists(model_path):
        print(f"成功加载训练好的模型: {model_path}")
        agent.load_model(model_path)
        agent.epsilon = 0.0  # 无探索
    else:
        print(f"警告: 模型文件不存在 - {model_path}")
        print("将使用随机策略进行演示")
        agent.epsilon = 1.0  # 完全随机

    print("\n" + "=" * 50)
    print("DQN贪吃蛇演示模式")
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"演示回合数: {episodes}")
    print("按任意键开始演示，按ESC退出")
    print("=" * 50 + "\n")

    # 显示初始状态
    state = env.reset()
    env.render()

    # 运行多个回合的演示
    scores = []
    rewards = []
    steps_list = []

    for episode in range(1, episodes + 1):
        print(f"\n开始演示回合 {episode}/{episodes}")

        state = env.reset()
        env.render()  # 确保显示初始状态
        total_reward = 0
        steps = 0

        while True:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            action =  agent.select_action(state,eval_model=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            # 渲染
            env.render()

            state = next_state

            # 检查是否结束
            if done:
                reason = info.get('termination', 'unknown')
                print(f"回合 {episode} 结束 - 原因: {reason}")
                # 短暂暂停显示结束状态
                for _ in range(10):
                    env.render()
                    pygame.time.wait(100)
                break

        # 记录结果
        scores.append(env.score)
        rewards.append(total_reward)
        steps_list.append(steps)

        print(f"回合 {episode} - 分数: {env.score}, 总奖励: {total_reward:.2f}, 步数: {steps}")

        # 短暂暂停，展示结果
        time.sleep(1.5)

    # 显示总结
    print("\n" + "=" * 50)
    print("演示结束 - 统计结果")
    print("=" * 50)
    print(f"平均分数: {np.mean(scores):.2f}")
    print(f"最高分数: {np.max(scores)}")
    print(f"平均总奖励: {np.mean(rewards):.2f}")
    print(f"平均步数: {np.mean(steps_list):.2f}")
    print("=" * 50)

    pygame.quit()


if __name__ == "__main__":

    model_path = "models/snake_dqn_final_noisy.pth"

    # 运行演示
    demo_trained_agent(model_path, episodes=5)