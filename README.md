# Noisy DQN 贪吃蛇

##  项目简介
使用 **Noisy DQN + Double Q-Learning** 在 20×20 网格环境中训练贪吃蛇智能体。用 Noisy Networks 替代传统 ε-greedy 实现自适应探索，手工设计 16 维状态特征（危险检测、食物方向、边界感知）
videos有项目演示视频

##  快速运行
\`\`\`bash
pip install -r requirements.txt
python demo.py --model_path models/snake_dqn_final_noisy.pth
\`\`\`

##  技术栈
- Python, PyTorch, Pygame
- Noisy Networks, Double DQN, Experience Replay
