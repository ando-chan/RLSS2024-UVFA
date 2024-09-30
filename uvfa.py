import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import csv
from decimal import Decimal, ROUND_UP

# デバイスの設定（GPUが利用可能であればGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# カスタム迷路環境の定義
class CustomMazeEnv(gym.Env):
    def __init__(self):
        super(CustomMazeEnv, self).__init__()

        self.grid_size = 15  # 迷路のサイズを15x15に設定

        # 迷路の定義（1が壁、0が通路）
        maze_str = """
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 0 1 0 0 0 0 0 0 0 0 0 0 0 1
        1 0 1 0 1 1 1 1 1 0 1 0 1 1 1
        1 0 1 0 1 0 0 0 0 0 1 0 1 1 1
        1 0 0 0 1 0 1 1 1 1 1 0 1 1 1
        1 1 1 0 0 0 0 0 0 0 1 1 0 1 1
        1 0 0 0 1 1 1 0 1 1 1 1 0 1 1
        1 0 1 0 1 2 0 0 0 0 0 1 0 1 1
        1 0 1 0 1 1 1 1 1 1 0 1 0 1 1
        1 0 1 0 0 0 0 0 1 0 0 0 0 1 1
        1 0 1 1 1 1 1 1 1 1 1 1 0 1 1
        1 0 1 0 0 0 0 0 0 0 1 0 0 0 1
        1 0 1 0 1 1 1 1 1 0 1 0 1 0 1
        1 0 0 0 0 0 0 0 1 0 0 0 1 0 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        """

        # 迷路データをパースしてNumPy配列に変換
        self.maze = self._parse_maze(maze_str)

        # 通行可能なセル（0のセル）の座標リストを作成
        self.free_cells = list(zip(*np.where(self.maze == 0)))

        # スタート位置とゴール位置を通行可能なセルから設定
        self.state = self._get_random_free_position()
        self.goal = self._get_random_free_position()

        self.action_space = spaces.Discrete(4)  # 4方向（上、下、左、右）
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.float32
        )

    def _parse_maze(self, maze_str):
        # 文字列から迷路データをパース
        maze_lines = maze_str.strip().split('\n')
        maze = []
        for line in maze_lines:
            maze.append([int(x) for x in line.strip().split()])
        return np.array(maze, dtype=int)

    def _get_random_free_position(self):
        # 通行可能なセルからランダムに位置を選択
        return list(random.choice(self.free_cells))

    def reset(self):
        # スタート位置とゴール位置を再設定
        self.state = [7, 5]
        return np.array(self.state, dtype=np.float32)

    def step(self, action, goal):
        x, y = self.state

        # 仮の次の位置を計算
        if action == 0:  # 上
            next_x = x - 1
            next_y = y
        elif action == 1:  # 下
            next_x = x + 1
            next_y = y
        elif action == 2:  # 左
            next_x = x
            next_y = y - 1
        elif action == 3:  # 右
            next_x = x
            next_y = y + 1
        else:
            raise ValueError("Invalid action")

        # 次の位置が壁でないか確認
        done = False
        if self.maze[next_x, next_y] == 0:
            self.state = [next_x, next_y]  # 壁でなければ位置を更新
            if self.state == goal:
                reward = 10.0  # ゴールで+1、それ以外は-0.1のペナルティ
                done = True
            else:
                reward = -0.1

        else:
            reward = -1.0  # 壁で-0.1のペナルティ

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        grid = np.copy(self.maze)
        x, y = self.state
        grid[x, y] = 3    # エージェントの位置を3に

        print(grid)

# UVFAのネットワーク定義（連続状態用）
class UVFANetwork(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(UVFANetwork, self).__init__()
        self.embd = nn.Embedding(15, 64)
        self.fc1 = nn.Linear(64*2 + 64*2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, action_dim)

    def forward(self, state, goal):
        state_x = self.embd(state[:,0])
        state_y = self.embd(state[:,1])
        state = torch.cat([state_x, state_y], dim=-1)
        goal_x = self.embd(goal[:, 0])
        goal_y = self.embd(goal[:, 1])
        goal = torch.cat([goal_x, goal_y], dim=-1)

        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.fc1(x))
        res = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) + res
        res = x
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x)) + res
        res = x
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x)) + res
        return self.fc8(x)

# 環境のセットアップ
env = CustomMazeEnv()
state_dim = env.observation_space.shape[0]  # 状態の次元（2）
goal_dim = state_dim  # ゴールも2次元
action_dim = env.action_space.n  # 行動数（4）

# UVFAネットワークとオプティマイザーの設定
uvfa = UVFANetwork(state_dim, goal_dim, action_dim).to(device)
optimizer = optim.Adam(uvfa.parameters(), lr=5e-5)
loss_fn = nn.MSELoss()

# 経験メモリの初期化
memory = deque(maxlen=5000)

# Q学習のパラメータ設定
gamma = 0.90  # 割引率
epsilon = 1.0  # ε-グリーディ戦略のε
epsilon_decay = 0.975
epsilon_min = 0.01
batch_size = 64

#goals = [[13, 7], [13, 13], [1, 1], [4, 11], [9, 7], [1, 13], [5, 12]]    

# 学習のメインループ
for episode in range(3000):
    state = env.reset()

    done = False
    total_reward = 0

    #goal = goals[np.random.randint(0, len(goals))]
    while 1:
        x = random.randint(1, 14)
        y = random.randint(1, 14)

        if env.maze[x][y] == 0:
            goal = [x, y]
            break

    c = 0
    while not done:
        # ε-グリーディで行動を選択
        if np.random.rand() < epsilon:
            action = np.random.randint(action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).to(torch.int64).to(device)
                goal_tensor = torch.tensor(goal).to(torch.int64).to(device)

                q_values = uvfa(state_tensor.unsqueeze(0), goal_tensor.unsqueeze(0))
                action = q_values.argmax().item()

        # 行動を実行して次の状態と報酬を取得
        next_state, reward, done, _ = env.step(action, goal)
        total_reward += reward

        # メモリに保存
        memory.append((state, goal, action, reward, next_state, done))

        state = next_state

        # 学習ステップ
        if len(memory) > batch_size and c % 10 == 0:
            batch = random.sample(memory, batch_size)
            state_batch = torch.tensor(np.array([x[0] for x in batch])).to(torch.int64).to(device)
            goal_batch = torch.tensor(np.array([x[1] for x in batch])).to(torch.int64).to(device)
            action_batch = torch.tensor(np.array([x[2] for x in batch])).to(torch.float32).to(device)
            reward_batch = torch.tensor(np.array([x[3] for x in batch])).to(torch.float32).to(device)
            next_state_batch = torch.tensor(np.array([x[4] for x in batch])).to(torch.int64).to(device)
            done_batch = torch.tensor(np.array([x[5] for x in batch])).to(dtype=torch.float32).to(device)

            # Q値のターゲット計算
            with torch.no_grad():
                next_q_values = uvfa(next_state_batch, goal_batch)
                max_next_q_values = next_q_values.max(dim=1)[0]
                target_q_values = reward_batch + (1 - done_batch) * gamma * max_next_q_values

            # 予測Q値の計算
            current_q_values = uvfa(state_batch, goal_batch).gather(1, action_batch.unsqueeze(1).to(torch.int64)).squeeze()

            # 損失を計算してバックプロパゲーション
            loss = loss_fn(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if c > 1000:
            break

        c += 1

    # 100エピソードごとに評価
    """
    if episode % 10 == 0 and episode != 0:
        epsilon *= epsilon_decay
        total_reward = 0
        step = [0, 0, 0, 0, 0, 0, 0]
        for test_episode in range(10):
            state = env.reset()
            goal = goals[test_episode]
            done = False

            count = 0
            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(state).to(torch.int64).to(device)
                    goal_tensor = torch.tensor(goal).to(torch.int64).to(device)

                    q_values = uvfa(state_tensor.unsqueeze(0), goal_tensor.unsqueeze(0))
                    action = q_values.argmax().item()

                state, reward, done, _ = env.step(action, goal)
                total_reward += reward
                step[test_episode] += 1

                if count == 100:
                    step[test_episode] = -1
                    break

                count += 1

        print(f"エピソード: {episode}, 平均総報酬: {total_reward/10}, ステップ:{step}")
    """
    
    if episode % 10 == 0 and episode != 0:
        epsilon *= epsilon_decay

        try_count = 0 
        succsess_count = 0
        for x in range(1, 15):
            for y in range(1, 15):
                if env.maze[x][y] == 0:
                    state = env.reset()
                    goal = [x, y]
                    try_count += 1
                    count = 0
                    done = False
                    while not done:
                        with torch.no_grad():
                            state_tensor = torch.tensor(state).to(torch.int64).to(device)
                            goal_tensor = torch.tensor(goal).to(torch.int64).to(device)

                            q_values = uvfa(state_tensor.unsqueeze(0), goal_tensor.unsqueeze(0))
                            action = q_values.argmax().item()

                        state, reward, done, _ = env.step(action, goal)

                        if count == 100:
                            break

                        if done:
                            succsess_count += 1

                        count += 1

        print(f"エピソード: {episode}, 到達率: {succsess_count/try_count}")
        p_list = [succsess_count/try_count]
        rounded_p_list = [[Decimal(str(p)).quantize(Decimal('0.001'), rounding=ROUND_UP)] for p in p_list]
        with open('/root/success_rates','a',newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(rounded_p_list)
print("学習完了")
env.close()