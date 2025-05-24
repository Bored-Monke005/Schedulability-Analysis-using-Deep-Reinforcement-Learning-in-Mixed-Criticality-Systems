# AMC-DQN Mixed-Criticality Scheduling Simulator with GUI, Gantt Highlighting, and Deadline Miss Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import gym
from gym import spaces
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Environment definition
class AMCSchedulingEnv(gym.Env):
    def __init__(self, task_set):
        super().__init__()
        self.task_set = task_set
        self.num_tasks = len(task_set)
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(self.num_tasks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_tasks, 2), dtype=np.float32)
        self.reset()

    def reset(self):
        self.budgets = np.array([t['lo_wcet'] for t in self.task_set], dtype=np.float32)
        self.execution = np.random.uniform(0.8, 1.2, size=self.num_tasks) * self.budgets
        return self._get_state()

    def _get_state(self):
        state = np.stack([self.budgets, self.execution], axis=1)
        return state / np.max(state)

    def step(self, action):
        self.budgets = np.clip(self.budgets + action * self.budgets, 0, [t['deadline'] for t in self.task_set])
        overrun = self.execution > self.budgets
        hi_mode_triggered = any(overrun & (np.array([t['criticality'] for t in self.task_set]) == 1))
        missed_deadlines = np.sum(self.execution > [t['deadline'] for t in self.task_set])
        reward = -missed_deadlines - hi_mode_triggered * 5
        done = True
        missed_indices = [i for i, exe in enumerate(self.execution) if exe > self.task_set[i]['deadline']]
        return self._get_state(), reward, done, {'misses': int(missed_deadlines), 'missed_indices': missed_indices}

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def __len__(self):
        return len(self.buffer)

# Training loop with live reward and Gantt chart updates
class DQNTrainer:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[0]
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(10000)
        self.epsilon = 1.0
        self.gamma = 0.99
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.episode = 0
        self.rewards = []
        self.misses = []
        self.total_misses = 0
        self.last_action = np.zeros(self.action_dim)
        self.last_missed_indices = []

    def train_step(self):
        state = self.env.reset().flatten()
        self.last_action = self.env.action_space.sample() if random.random() < self.epsilon else self.policy_net(torch.FloatTensor(state)).detach().numpy()
        next_state, reward, _, info = self.env.step(self.last_action)
        self.buffer.push(state, self.last_action, reward, next_state.flatten())
        self.rewards.append(reward)
        self.misses.append(info['misses'])
        self.total_misses += info['misses']
        self.last_missed_indices = info['missed_indices']

        if len(self.buffer) >= self.batch_size:
            states, actions, rewards_batch, next_states = self.buffer.sample(self.batch_size)
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards_batch)
            next_states = torch.FloatTensor(next_states)

            q_values = self.policy_net(states)
            next_q_values = self.target_net(next_states).detach()
            expected_q_value = rewards_tensor + self.gamma * next_q_values.max(1)[0]

            loss = nn.MSELoss()(q_values, expected_q_value.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.episode % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon *= self.epsilon_decay
        self.episode += 1
        return reward, info['misses']

# GUI
class SimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AMC-DQN Scheduler Simulator")
        self.root.geometry("900x700")

        self.theme = 'light'
        self._set_theme()

        # Style setup
        self.style = ttk.Style()
        self._apply_style()

        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.top_frame = ttk.Frame(self.frame)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = ttk.Button(self.top_frame, text="Load Task File", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.pause_button = ttk.Button(self.top_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(self.top_frame, text="Reset", command=self.reset_simulation, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.theme_button = ttk.Button(self.top_frame, text="Toggle Theme", command=self.toggle_theme)
        self.theme_button.pack(side=tk.LEFT, padx=5)

        self.label_misses = ttk.Label(self.top_frame, text="Deadline Misses: 0 | Total Misses: 0", font=("Segoe UI", 10, "bold"))
        self.label_misses.pack(side=tk.RIGHT, padx=5)

        self.canvas_frame = ttk.Frame(self.frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax_reward, self.ax_gantt) = plt.subplots(2, 1, figsize=(9, 6))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.env = None
        self.trainer = None
        self.paused = False
        self.after_id = None

    def _set_theme(self):
        bg_color = "#121212" if self.theme == 'dark' else "#f0f0f0"
        self.root.configure(bg=bg_color)

    def _apply_style(self):
        self.style.theme_use('clam')
        if self.theme == 'dark':
            self.style.configure("TFrame", background="#121212")
            self.style.configure("TLabel", background="#121212", foreground="white")
            self.style.configure("TButton", background="#333", foreground="white")
        else:
            self.style.configure("TFrame", background="#f0f0f0")
            self.style.configure("TLabel", background="#f0f0f0", foreground="black")
            self.style.configure("TButton", background="#e0e0e0", foreground="black")

    def toggle_theme(self):
        self.theme = 'dark' if self.theme == 'light' else 'light'
        self._set_theme()
        self._apply_style()

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Input Files", "*.in"), ("All Files", "*.*")])
        if not filepath:
            return
        task_set = self.parse_task_file(filepath)
        self.env = AMCSchedulingEnv(task_set)
        self.trainer = DQNTrainer(self.env)
        self.pause_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.paused = False
        self.update_plot()

    def update_plot(self):
        if self.trainer and not self.paused:
            reward, misses = self.trainer.train_step()
            self.ax_reward.clear()
            self.ax_reward.plot(self.trainer.rewards[-50:], label="Rewards", color='purple')
            self.ax_reward.set_title(f"Episode {self.trainer.episode} - Reward: {reward:.2f}", fontsize=11)
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.set_xlabel("Episodes")
            self.ax_reward.grid(True)
            self.ax_reward.legend()

            self.ax_gantt.clear()
            time = 0
            for i, task in enumerate(self.env.task_set):
                if i in self.trainer.last_missed_indices:
                    edgecolor = 'black'
                    hatch = '//'
                else:
                    edgecolor = 'none'
                    hatch = ''

                if task['criticality'] == 1:
                    color = 'red' if self.env.execution[i] > self.env.budgets[i] else 'blue'
                else:
                    color = 'orange' if self.env.execution[i] > self.env.budgets[i] else 'green'

                self.ax_gantt.broken_barh(
                    [(time, task['exec_time'])],
                    (i * 10, 9),
                    facecolors=color,
                    edgecolors=edgecolor,
                    hatch=hatch
                )
                self.ax_gantt.text(time + 1, i * 10 + 4, f"T{i}", va='center', fontsize=8)
                time += task['exec_time']

            self.ax_gantt.set_title("Task Execution Timeline", fontsize=11)
            self.ax_gantt.set_xlabel("Time")
            self.ax_gantt.set_yticks([])
            self.ax_gantt.grid(True, linestyle='--', alpha=0.5)

            self.label_misses.config(text=f"Deadline Misses: {misses} | Total Misses: {self.trainer.total_misses}")
            self.canvas.draw()
        self.after_id = self.root.after(500, self.update_plot)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def reset_simulation(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.ax_reward.clear()
        self.ax_gantt.clear()
        self.canvas.draw()
        self.label_misses.config(text="Deadline Misses: 0 | Total Misses: 0")
        self.trainer = None
        self.pause_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)

    def parse_task_file(self, filename):
        task_set = []
        with open(filename) as f:
            for line in f:
                if line.startswith("---"): break
                parts = list(map(int, line.strip().split()))
                if len(parts) == 5:
                    task_set.append({
                        'exec_time': parts[0],
                        'lo_wcet': parts[1],
                        'hi_wcet': parts[2],
                        'criticality': parts[3],
                        'deadline': parts[4]
                    })
        return task_set

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorApp(root)
    root.mainloop()
