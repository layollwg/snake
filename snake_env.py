import numpy as np
import random


class SnakeEnv:
    """
    贪吃蛇游戏环境 — 用于 Q-learning 强化学习项目。

    坐标系统：
        (row, col), (0,0) 是左上角，row 向下增长，col 向右增长。

    绝对方向 (absolute direction)：
        0 = 上(UP), 1 = 下(DOWN), 2 = 左(LEFT), 3 = 右(RIGHT)

    相对动作 (relative action, 蛇不能直接 180° 掉头)：
        0 = 直行(straight)    → 保持当前方向
        1 = 左转(turn left)   → 基于当前方向左转
        2 = 右转(turn right)  → 基于当前方向右转

    状态表示 — 基础模式 (11 维二进制特征, 共 2^11 = 2048 个状态)：
        [ds, dl, dr, dir_u, dir_d, dir_l, dir_r, fu, fd, fl, fr]
           └ 危险 ┘  └──── 当前方向(one-hot) ────┘  └─ 食物方向 ─┘
        ds/dl/dr: 直行/左转/右转是否会立即碰撞 (1=危险)
        dir_*:    当前绝对方向 (恰好一位为 1)
        fu/fd/fl/fr: 食物相对于蛇头的位置 (至多一位为 1)

    增强状态模式 (use_enhanced_state=True, 15 维, 共 2^15 = 32768 个状态)：
        在基础 11 位之后额外追加 4 位：
        [d2_ss, d2_ls, d2_rs, len_long]
        d2_ss: 直行→直行 两步后是否碰墙 (1=危险)
        d2_ls: 左转→直行 两步后是否碰墙 (1=危险)
        d2_rs: 右转→直行 两步后是否碰墙 (1=危险)
        len_long: 蛇身长度是否超过 grid_size (1=长蛇)

    奖励函数：
        +10  吃到食物
        -10  死亡 (撞墙 / 撞自己)
        -0.1 普通每一步 (鼓励蛇尽快吃食物)
    """

    def __init__(self, grid_size=6, initial_length=2, max_steps=1000, use_enhanced_state=False):
        """
        参数：
            grid_size: 棋盘大小 (默认 6×6, 也可设 8×8 做难度对比)
            initial_length: 蛇初始长度 (默认 2, 即 1 头 + 1 节身体)
            max_steps: 单局最大步数, 防止无限循环
            use_enhanced_state: 是否启用 15 位增强状态编码 (额外 3 个 2步预判危险位 + 1 个蛇身长度位)
        """
        if grid_size < 4:
            raise ValueError("棋盘至少需要 4×4")
        if initial_length < 1 or initial_length > grid_size * grid_size // 2:
            raise ValueError("initial_length 不合理")

        self.grid_size = grid_size
        self.initial_length = initial_length
        self.max_steps = max_steps
        self.use_enhanced_state = use_enhanced_state
        # 基础: 2^11 = 2048; 增强: 2^15 = 32768 (额外 4 位)
        self.num_states = 32768 if use_enhanced_state else 2048
        self.num_actions = 3

        self.action_names = {0: "straight", 1: "turn_left", 2: "turn_right"}
        self.dir_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        self.reset()

    def reset(self):
        """
        重置一局 (episode)：清空计数器 → 蛇放中心 → 随机初始方向 →
        在反方向生成蛇身 → 随机放食物 → 返回初始 state。
        """
        self.step_count = 0
        self.total_reward = 0.0
        self.score = 0

        center = self.grid_size // 2

        self.direction = random.choice([0, 1, 2, 3])

        # 反方向映射: 用于在头的"后面"生成蛇身
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}

        self.body = [None] * self.initial_length
        self.body[0] = [center, center]        # 蛇头放中心

        # 身体沿着反方向依次排列
        # 反方向的位移量 (dr, dc): UP→向下(down)位移(1,0), DOWN→向上(-1,0), etc.
        dir_offsets = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
        dr, dc = dir_offsets[opposite[self.direction]]

        for i in range(1, self.initial_length):
            self.body[i] = [self.body[i - 1][0] + dr,
                            self.body[i - 1][1] + dc]

        self._occupied = set()
        for seg in self.body:
            self._occupied.add((seg[0], seg[1]))

        self._place_food()

        return self.get_state()

    def _place_food(self):
        """在所有未被蛇占据的格子中随机放置食物。棋盘满了则 food=None。"""
        available = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self._occupied:
                    available.append((r, c))

        if not available:
            self.food = None
        else:
            self.food = list(random.choice(available))

    def _get_new_direction(self, action):
        """
        根据相对动作计算新的绝对方向。

        转向规则 (以上为例): straight→仍是上, turn_left→左, turn_right→右。
        四个方向的左转/右转通过查表实现。
        """
        if action == 0:
            return self.direction

        LEFT_TURN = {0: 2, 1: 3, 2: 1, 3: 0}
        RIGHT_TURN = {0: 3, 1: 2, 2: 0, 3: 1}

        if action == 1:
            return LEFT_TURN[self.direction]
        elif action == 2:
            return RIGHT_TURN[self.direction]
        else:
            raise ValueError(f"无效动作: {action}")

    def _move_head_position(self, head, direction):
        """根据当前方向和蛇头位置，计算移动后的新 (row, col)。"""
        r, c = head
        if direction == 0:           # UP
            return (r - 1, c)
        elif direction == 1:         # DOWN
            return (r + 1, c)
        elif direction == 2:         # LEFT
            return (r, c - 1)
        elif direction == 3:         # RIGHT
            return (r, c + 1)

    def _is_collision(self, position, grow=False):
        """
        判断 position 是否发生碰撞。
        - 超出棋盘边界 → 撞墙
        - 与蛇身重叠 → 撞自己
        grow=True 表示这一帧吃到了食物，尾巴不移走，所以蛇尾也是障碍。
        """
        r, c = position
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True

        body_set = set((s[0], s[1]) for s in self.body)
        if not grow:
            tail = self.body[-1]
            body_set.discard((tail[0], tail[1]))

        return (r, c) in body_set

    def _would_collide(self, action):
        """试探: 如果执行 action 会立刻碰撞吗？(不真的改变蛇位置)"""
        new_dir = self._get_new_direction(action)
        new_head = self._move_head_position(self.body[0], new_dir)
        return self._is_collision(new_head)

    def _would_collide_2step(self, action1: int, action2: int = 0) -> bool:
        """两步预判：先执行 action1，再执行 action2，最终是否碰撞？

        第一步做完整碰撞检测（墙 + 蛇身），第二步仅检测边界碰撞（蛇身位置
        难以精确模拟，但边界信息已足够帮助智能体规避死角）。
        """
        # --- 第一步 ---
        new_dir1 = self._get_new_direction(action1)
        new_head1 = self._move_head_position(self.body[0], new_dir1)
        if self._is_collision(new_head1):
            return True  # 第一步已撞

        # --- 第二步方向（相对于 new_dir1） ---
        LEFT_TURN = {0: 2, 1: 3, 2: 1, 3: 0}
        RIGHT_TURN = {0: 3, 1: 2, 2: 0, 3: 1}
        if action2 == 0:
            new_dir2 = new_dir1
        elif action2 == 1:
            new_dir2 = LEFT_TURN[new_dir1]
        else:
            new_dir2 = RIGHT_TURN[new_dir1]

        r, c = self._move_head_position(new_head1, new_dir2)
        # 仅检测边界（蛇身二步预判忽略，保持计算简洁）
        return r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size

    def get_state(self):
        """
        将当前棋盘状态编码为二进制整数。

        基础模式 (use_enhanced_state=False): 11 位, 0~2047。
        增强模式 (use_enhanced_state=True):  15 位, 0~32767。

        位布局（从高位到低位）：
            0: ds    — 直行会碰撞？
            1: dl    — 左转会碰撞？
            2: dr    — 右转会碰撞？
            3: dir_u — 方向 = 上
            4: dir_d — 方向 = 下
            5: dir_l — 方向 = 左
            6: dir_r — 方向 = 右
            7: fu    — 食物在上方
            8: fd    — 食物在下方
            9: fl    — 食物在左方
            10: fr   — 食物在右方
        增强模式额外 4 位（仅 use_enhanced_state=True）：
            11: d2_ss  — 直行再直行 两步后碰墙？
            12: d2_ls  — 左转再直行 两步后碰墙？
            13: d2_rs  — 右转再直行 两步后碰墙？
            14: len_long — 蛇身长度 > grid_size？
        """
        state_bits = [0] * 11

        # 前三位: 危险信号
        state_bits[0] = 1 if self._would_collide(0) else 0
        state_bits[1] = 1 if self._would_collide(1) else 0
        state_bits[2] = 1 if self._would_collide(2) else 0

        # 中间四位: 当前方向 (one-hot)
        state_bits[3] = 1 if self.direction == 0 else 0
        state_bits[4] = 1 if self.direction == 1 else 0
        state_bits[5] = 1 if self.direction == 2 else 0
        state_bits[6] = 1 if self.direction == 3 else 0

        # 最后四位: 食物相对位置
        if self.food is not None:
            head_r, head_c = self.body[0]
            food_r, food_c = self.food

            state_bits[7] = 1 if food_r < head_r else 0   # 食物在上
            state_bits[8] = 1 if food_r > head_r else 0   # 食物在下
            state_bits[9] = 1 if food_c < head_c else 0   # 食物在左
            state_bits[10] = 1 if food_c > head_c else 0  # 食物在右

        # 增强模式: 追加 4 个额外位
        if self.use_enhanced_state:
            state_bits.append(1 if self._would_collide_2step(0, 0) else 0)  # 直行→直行
            state_bits.append(1 if self._would_collide_2step(1, 0) else 0)  # 左转→直行
            state_bits.append(1 if self._would_collide_2step(2, 0) else 0)  # 右转→直行
            state_bits.append(1 if len(self.body) > self.grid_size else 0)  # 长蛇标志

        # bits → 整数
        state_int = 0
        for bit_val in state_bits:
            state_int = (state_int << 1) | bit_val

        return state_int

    def get_state_vector(self):
        """返回状态的 numpy 数组 (调试/可视化用)。长度随模式不同为 11 或 15。"""
        state_int = self.get_state()
        n_bits = 15 if self.use_enhanced_state else 11
        bits = [(state_int >> i) & 1 for i in range(n_bits - 1, -1, -1)]
        return np.array(bits, dtype=np.float32)

    def step(self, action):
        """
        执行一步环境交互 (类似 OpenAI Gym 的 step)。

        参数：action ∈ {0, 1, 2}
        返回：(state, reward, done, info)

        流程：更新方向 → 算新位置 → 判断吃食物/碰撞 → 更新蛇身 → 检查终止。
        """
        if action not in (0, 1, 2):
            raise ValueError(f"动作必须在 0/1/2 之间, 收到: {action}")

        self.step_count += 1

        self.direction = self._get_new_direction(action)

        new_head = list(
            self._move_head_position(self.body[0], self.direction)
        )

        # 判断是否吃到食物
        grew = False
        if self.food is not None and new_head == self.food:
            grew = True
            self.score += 1
            reward = 10.0
        else:
            reward = -0.1

        # 碰撞检测 (吃到食物时 tail 不移走, grow=True)
        if self._is_collision(new_head, grow=grew):
            reward = -10.0
            self.total_reward += reward
            done = True
            return self.get_state(), reward, done, {
                "score": self.score,
                "total_reward": self.total_reward,
                "steps": self.step_count,
                "cause": "collision",
            }

        # 插入新头部
        self.body.insert(0, new_head)
        self._occupied.add((new_head[0], new_head[1]))

        if grew:
            # 吃到食物: 不删尾巴 (变长), 生成新食物
            self.food = None
            self._place_food()
        else:
            # 没吃到: 删除尾巴 (保持原长)
            tail = self.body.pop()
            self._occupied.discard((tail[0], tail[1]))

        self.total_reward += reward

        done = self.step_count >= self.max_steps

        return self.get_state(), reward, done, {
            "score": self.score,
            "total_reward": self.total_reward,
            "steps": self.step_count,
            "cause": "max_steps" if done else "",
        }

    def render(self):
        """
        在控制台打印棋盘。
        H = 蛇头, S = 蛇身, F = 食物, . = 空格
        """
        grid = [["." for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]

        if self.food is not None:
            fr, fc = self.food
            grid[fr][fc] = "F"

        for i, seg in enumerate(self.body):
            r, c = seg
            grid[r][c] = "H" if i == 0 else "S"

        print("-" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.grid_size * 2 + 1))
        print(f"Score: {self.score} | Steps: {self.step_count} | "
              f"Direction: {self.dir_names[self.direction]}")

    def get_info(self):
        """返回环境元数据。"""
        return {
            "grid_size": self.grid_size,
            "initial_length": self.initial_length,
            "max_steps": self.max_steps,
            "use_enhanced_state": self.use_enhanced_state,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "action_names": self.action_names,
        }
