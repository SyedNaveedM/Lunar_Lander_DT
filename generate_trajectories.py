# import gymnasium as gym
# import numpy as np
# import torch
# import os
# import pickle
# import warnings
# import time
# from tqdm import tqdm
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# from stable_baselines3.common.vec_env import SubprocVecEnv

# warnings.filterwarnings('ignore')

# # === Helper for parallel environment creation ===
# def make_env():
#     def _init():
#         env = gym.make("LunarLander-v3")
#         return env
#     return _init

# # === Enhanced Progress bar callback with detailed metrics ===
# class EnhancedProgressCallback(BaseCallback):
#     def __init__(self, check_freq=1000, verbose=0):
#         super().__init__(verbose)
#         self.check_freq = check_freq
#         self.pbar = None
#         self.start_time = None
#         self.last_time = None
#         self.last_steps = 0
#         self.episode_rewards = []
#         self.episode_count = 0

#     def _on_training_start(self) -> None:
#         self.start_time = time.time()
#         self.last_time = self.start_time
#         total_steps = self.locals.get('total_timesteps', 1_000_000)
#         self.pbar = tqdm(
#             total=total_steps,
#             desc="\U0001F3CB️ Training PPO",
#             unit="steps",
#             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
#         )

#     def _on_step(self) -> bool:
#         if 'infos' in self.locals:
#             for info in self.locals['infos']:
#                 if 'episode' in info:
#                     reward = info['episode']['r']
#                     self.episode_rewards.append(reward)
#                     self.episode_count += 1
#                     if len(self.episode_rewards) > 100:
#                         self.episode_rewards.pop(0)

#         if self.n_calls % self.check_freq == 0:
#             now = time.time()
#             steps = self.num_timesteps - self.last_steps
#             dt = now - self.last_time
#             sps = steps / dt if dt > 0 else 0

#             postfix = {"SPS": f"{sps:.0f}"}
#             if self.episode_rewards:
#                 avg_reward = np.mean(self.episode_rewards[-20:])
#                 max_reward = np.max(self.episode_rewards[-20:])
#                 success_rate = np.mean(np.array(self.episode_rewards[-20:]) >= 200) * 100
#                 postfix.update({
#                     "AvgR": f"{avg_reward:.0f}",
#                     "MaxR": f"{max_reward:.0f}",
#                     "Success": f"{success_rate:.0f}%"
#                 })

#             self.last_steps = self.num_timesteps
#             self.last_time = now

#             if self.pbar:
#                 self.pbar.n = self.num_timesteps
#                 self.pbar.set_postfix(postfix)
#                 self.pbar.refresh()
#         return True

#     def _on_training_end(self) -> None:
#         if self.pbar:
#             self.pbar.n = self.pbar.total
#             elapsed = time.time() - self.start_time
#             final_stats = f"Episodes: {self.episode_count}, Time: {elapsed:.1f}s"
#             if self.episode_rewards:
#                 avg_final = np.mean(self.episode_rewards[-50:])
#                 final_stats += f", Final Avg: {avg_final:.1f}"
#             self.pbar.set_description(f"\u2705 Training Complete")
#             self.pbar.set_postfix_str(final_stats)
#             self.pbar.refresh()
#             self.pbar.close()

# # === Performance testing function ===
# def quick_performance_test(model, env, test_episodes=20):
#     print("\U0001F9EA Performance Test...")
#     returns = []
#     success_count = 0
    
#     test_pbar = tqdm(range(test_episodes), desc="Testing", 
#                     bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| [{postfix}]')

#     for i in test_pbar:
#         obs, _ = env.reset()
#         done = False
#         total_return = 0

#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             total_return += reward

#         returns.append(total_return)
#         if total_return >= 250:
#             success_count += 1

#         avg_return = np.mean(returns)
#         success_rate = (success_count / (i + 1)) * 100
#         test_pbar.set_postfix({
#             "Avg": f"{avg_return:.0f}",
#             "Success": f"{success_rate:.0f}%",
#             "Last": f"{total_return:.0f}"
#         })

#     test_pbar.close()
#     return returns, success_count / test_episodes

# def generate_expert_trajectories(env_name="LunarLander-v3", n_episodes=2500, return_threshold=250):
#     print("\U0001F680 Enhanced Expert Trajectory Generation")
#     print("=" * 55)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\U0001F527 Device: {device}")

#     n_envs = 15 if device.type == "cuda" else 4
#     print(f"\U0001F527 Parallel environments: {n_envs}")

#     vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
#     eval_env = Monitor(gym.make(env_name))
#     single_env = gym.make(env_name)

#     model = PPO(
#         "MlpPolicy",
#         vec_env,
#         learning_rate=3e-4,
#         n_steps=1024,
#         batch_size=128,
#         n_epochs=8,
#         gamma=0.999,
#         gae_lambda=0.98,
#         clip_range=0.2,
#         ent_coef=0.005,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=0,
#         device=device,
#         policy_kwargs=dict(
#             net_arch=[dict(pi=[128, 128], vf=[128, 128])],
#             activation_fn=torch.nn.ReLU
#         )
#     )

#     eval_callback = EvalCallback(
#         eval_env, 
#         best_model_save_path="./models/",
#         log_path="./logs/",
#         eval_freq=3000,
#         deterministic=True,
#         render=False,
#         n_eval_episodes=10,
#         verbose=0
#     )
#     progress_callback = EnhancedProgressCallback(check_freq=500)

#     print("\n\U0001F4C8 Training PPO Agent for High Performance...")
#     start_time = time.time()

#     try:
#         model.learn(
#             total_timesteps=2_000_000,
#             callback=[eval_callback, progress_callback],
#             progress_bar=False
#         )
#     except Exception as e:
#         print(f"\n⏹️ Training stopped: {e}")

#     training_time = time.time() - start_time
#     print(f"\n⏱️ Training completed in {training_time:.1f} seconds")

#     best_model_path = "./models/best_model.zip"
#     if os.path.exists(best_model_path):
#         print("📂 Loading best model...")
#         model = PPO.load(best_model_path, env=single_env, device=device)
#     else:
#         print("📂 Using final model...")
#         model.set_env(single_env)

#     os.makedirs("models", exist_ok=True)
#     model.save("models/ppo_lunarlander_expert")

#     test_returns, success_rate = quick_performance_test(model, single_env, test_episodes=25)
#     avg_return = np.mean(test_returns)
#     std_return = np.std(test_returns)

#     print(f"\n📊 Performance Summary:")
#     print(f"   Average Return: {avg_return:.1f} ± {std_return:.1f}")
#     print(f"   Success Rate (≥250): {success_rate*100:.1f}%")
#     print(f"   Max Return: {np.max(test_returns):.1f}")

#     actual_threshold = return_threshold

#     expert_trajectories = []
#     collected = 0
#     attempts = 0
#     recent_returns = []
#     collection_start = time.time()

#     np.random.seed(42)
#     seeds = np.random.randint(0, 1000000, size=n_episodes * 5)

#     print(f"\n📊 Collecting {n_episodes} Expert Trajectories (Return ≥ {actual_threshold:.0f})...")

#     collection_pbar = tqdm(
#         total=n_episodes, 
#         desc="\U0001F3AF Collecting Expert Data",
#         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
#     )

#     seed_idx = 0
#     consecutive_failures = 0
#     max_consecutive_failures = 100

#     while collected < n_episodes and seed_idx < len(seeds) and consecutive_failures < max_consecutive_failures:
#         single_env.reset(seed=int(seeds[seed_idx]))
#         obs, _ = single_env.reset()
#         done = False

#         traj = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}
#         total_return = 0

#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             next_obs, reward, terminated, truncated, _ = single_env.step(action)
#             done = terminated or truncated

#             traj["states"].append(obs.copy())
#             traj["actions"].append(action)
#             traj["rewards"].append(reward)
#             traj["next_states"].append(next_obs.copy())
#             traj["dones"].append(done)

#             obs = next_obs
#             total_return += reward

#         attempts += 1
#         seed_idx += 1
#         recent_returns.append(total_return)
#         if len(recent_returns) > 100:
#             recent_returns.pop(0)

#         if total_return >= actual_threshold:
#             expert_trajectories.append(traj)
#             collected += 1
#             consecutive_failures = 0
#             status = "✅"
#             collection_pbar.update(1)
#         else:
#             consecutive_failures += 1
#             status = "❌"

#         if attempts % 5 == 0 or collected != (collected // 5) * 5:
#             success_rate = (collected / attempts) * 100 if attempts > 0 else 0
#             avg_recent = np.mean(recent_returns[-20:]) if len(recent_returns) >= 20 else np.mean(recent_returns)
#             time_per_traj = (time.time() - collection_start) / attempts
#             eta_seconds = time_per_traj * (n_episodes - collected) / max(success_rate/100, 0.01)

#             collection_pbar.set_postfix({
#                 "Last": f"{status} {total_return:.0f}",
#                 "Avg20": f"{avg_recent:.0f}",
#                 "Success": f"{success_rate:.0f}%",
#                 "ETA": f"{eta_seconds/60:.1f}m"
#             })
#             collection_pbar.refresh()

#     collection_pbar.close()
#     vec_env.close()
#     eval_env.close() 
#     single_env.close()

#     collection_time = time.time() - collection_start
#     final_success_rate = (collected / attempts) * 100 if attempts > 0 else 0

#     print(f"\n\u2705 Collection Complete!")
#     print(f"   \U0001F4CA Collected: {collected}/{n_episodes} trajectories")
#     print(f"   \U0001F4CA Success rate: {final_success_rate:.1f}% ({collected}/{attempts} attempts)")
#     print(f"   ⏱️ Collection time: {collection_time:.1f} seconds")
#     print(f"   ⚡ Rate: {collected/(collection_time/60):.1f} trajectories/minute")

#     return expert_trajectories

# if __name__ == "__main__":
#     for dir_name in ["models", "logs", "datasets"]:
#         os.makedirs(dir_name, exist_ok=True)

#     print("\U0001F3AE LunarLander Expert Trajectory Generation")
#     print("=" * 55)

#     total_start = time.time()

#     dataset = generate_expert_trajectories(
#         n_episodes=5000,
#         return_threshold=250
#     )

#     dataset_path = "datasets/trajectories_5000.pkl"
#     with open(dataset_path, "wb") as f:
#         pickle.dump(dataset, f)

#     total_time = time.time() - total_start

#     print(f"\n\U0001F389 Mission Complete!")
#     print(f"💾 Saved {len(dataset)} trajectories to '{dataset_path}'")
#     print(f"⏱️ Total time: {total_time/60:.1f} minutes")
#     print(f"\U0001F680 Ready for imitation learning!")























import gymnasium as gym
import numpy as np
import torch
import os
import pickle
import warnings
import time
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings('ignore')

def make_env():
    return lambda: gym.make("LunarLander-v3")

class EnhancedProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.locals.get('total_timesteps', 1_000_000),
            desc="🏋y Training PPO",
            unit="steps",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
        )
        self.last_time = time.time()
        self.last_steps = 0
        self.rewards = []

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.rewards.append(info['episode']['r'])
                if len(self.rewards) > 100:
                    self.rewards.pop(0)

        if self.n_calls % self.check_freq == 0:
            now = time.time()
            dt = now - self.last_time
            steps = self.num_timesteps - self.last_steps
            sps = steps / dt if dt > 0 else 0
            avg_r = np.mean(self.rewards[-20:]) if self.rewards else 0
            max_r = np.max(self.rewards[-20:]) if self.rewards else 0
            success = np.mean(np.array(self.rewards[-20:]) >= 200) * 100 if self.rewards else 0

            self.pbar.n = self.num_timesteps
            self.pbar.set_postfix({"SPS": f"{sps:.0f}", "AvgR": f"{avg_r:.0f}", "MaxR": f"{max_r:.0f}", "Success": f"{success:.0f}%"})
            self.pbar.refresh()

            self.last_time, self.last_steps = now, self.num_timesteps
        return True

    def _on_training_end(self):
        self.pbar.n = self.pbar.total
        self.pbar.set_description("\u2705 Training Complete")
        self.pbar.set_postfix_str(f"Episodes: ~{len(self.rewards)}, Final Avg: {np.mean(self.rewards[-50:]):.1f}")
        self.pbar.close()

def quick_test(model, env, episodes=20):
    returns, success = [], 0
    for _ in tqdm(range(episodes), desc="Testing", bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| [{postfix}]'):
        obs, _ = env.reset()
        done, total_r = False, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(model.predict(obs, deterministic=True)[0])
            done, total_r = terminated or truncated, total_r + reward
        returns.append(total_r)
        success += total_r >= 250
    return returns, success / episodes

def generate_expert_trajectories(env_name="LunarLander-v3", n_episodes=2500, return_threshold=250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_envs = 12 if device.type == "cuda" else 4

    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = Monitor(gym.make(env_name))
    single_env = gym.make(env_name)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=1024, batch_size=128,
        n_epochs=8, gamma=0.999, gae_lambda=0.98, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        verbose=0, device=device,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])], activation_fn=torch.nn.ReLU)
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=[
            EvalCallback(eval_env, best_model_save_path="./models/", log_path="./logs/", eval_freq=3000,
                         deterministic=True, render=False, n_eval_episodes=10, verbose=0),
            EnhancedProgressCallback(check_freq=500)
        ],
        progress_bar=False
    )

    best_model_path = "./models/best_model.zip"
    model = PPO.load(best_model_path, env=single_env, device=device) if os.path.exists(best_model_path) else model.set_env(single_env)
    model.save("models/ppo_lunarlander_expert")

    test_returns, success_rate = quick_test(model, single_env, episodes=25)
    print(f"\nPerformance: Avg {np.mean(test_returns):.1f}, Max {np.max(test_returns):.1f}, Success {success_rate*100:.1f}%")

    expert_data, seeds = [], np.random.randint(0, 1e6, n_episodes * 5)
    collected, idx, fails, start = 0, 0, 0, time.time()

    pbar = tqdm(total=n_episodes, desc="\U0001F3AF Collecting Expert Data", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')

    while collected < n_episodes and idx < len(seeds) and fails < 100:
        single_env.reset(seed=int(seeds[idx]))
        obs, _ = single_env.reset()
        traj, done, ret = {k: [] for k in ["states", "actions", "rewards", "next_states", "dones"]}, False, 0

        while not done:
            action = model.predict(obs, deterministic=True)[0]
            next_obs, reward, terminated, truncated, _ = single_env.step(action)
            done = terminated or truncated
            for k, v in zip(traj.keys(), [obs.copy(), action, reward, next_obs.copy(), done]):
                traj[k].append(v)
            obs, ret = next_obs, ret + reward

        if ret >= return_threshold:
            expert_data.append(traj)
            collected += 1
            fails = 0
            pbar.update(1)
        else:
            fails += 1

        idx += 1
        if idx % 5 == 0:
            rate = collected / (time.time() - start) * 60
            pbar.set_postfix({"LastR": f"{ret:.0f}", "Success": f"{(collected / idx) * 100:.1f}%", "Rate": f"{rate:.1f}/min"})
            pbar.refresh()

    vec_env.close(), eval_env.close(), single_env.close()
    print(f"\n\u2705 Done: {collected} trajectories in {(time.time() - start):.1f}s")
    return expert_data

if __name__ == "__main__":
    for d in ["models", "logs", "datasets"]:
        os.makedirs(d, exist_ok=True)

    dataset = generate_expert_trajectories(n_episodes=5000, return_threshold=250)
    with open("datasets/trajectories_5000.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print("\n\U0001F389 All done! Ready for imitation learning.")