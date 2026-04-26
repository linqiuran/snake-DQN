import os
import numpy as np
import torch
import pygame
from snake_env import SnakeEnv
from dqn_agent import DQNAgent


def train_dqn(env, agent, num_episodes=1000, render_every=100):
    """训练DQN智能体"""
    print(f"开始训练，共 {num_episodes} 回合...")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0

        while True:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)

            # 优化模型
            loss = agent.optimize_model()
            episode_loss += loss

            # 更新状态
            state = next_state

            # 检查是否结束
            if done:
                break

        # 更新探索率
        # agent.update_epsilon()

        # 记录统计信息
        agent.episode_rewards.append(episode_reward)
        agent.episode_losses.append(episode_loss / max(1, step_count))
        agent.avg_sigma = agent.get_avg_sigma()
        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:])
            avg_loss = np.mean(agent.episode_losses[-10:])
            print(f"回合: {episode}/{num_episodes}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"平均损失: {avg_loss:.4f}, "
                  #f"探索率: {agent.epsilon:.3f}, "
                  f"avg:{agent.avg_sigma}",
                  f"分数: {env.score}")

        # 定期渲染
        if episode % render_every == 0:
            print(f"正在渲染第 {episode} 回合...")
            test_agent(env, agent, render=True)

        # 保存模型
        if episode % 5000 == 0:
            agent.save_model(f"models/snake_dqn_episode_{episode}_noisy.pth")

    print("训练完成!")
    agent.save_model("models/snake_dqn_final_noisy.pth")
    return agent


def test_agent(env, agent, render=True):
    """测试训练好的智能体"""
    state = env.reset()
    total_reward = 0
    steps = 0

    while True:
        # 选择动作 (不使用随机探索)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            q_values = agent.policy_net_noisy(state_tensor)
            action = q_values.argmax().item()

        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if render:
            env.render()

        state = next_state

        if done:
            break

    print(f"测试结果 - 总奖励: {total_reward:.2f}, 分数: {env.score}, 步数: {steps}")
    return total_reward, env.score, steps


if __name__ == "__main__":
    # 确保models目录存在
    os.makedirs("models", exist_ok=True)

    # 创建环境
    render_training = False  # 训练时是否渲染
    env = SnakeEnv(grid_width=20, grid_height=20, render=render_training)

    # 创建DQN智能体
    agent = DQNAgent(env)

    # 检查是否有预训练模型
    model_path = "models/snake_dqn_final_noisy.pth"
    if os.path.exists(model_path):


        print(f"加载预训练模型: {model_path}")
        agent.load_model(model_path)
        # agent.epsilon = 0.1

    # 训练
    train_episodes = 10000
    if train_episodes > 0:
        agent = train_dqn(env, agent, num_episodes=train_episodes, render_every=100)

    # 关闭Pygame
    pygame.quit()