"""
示例：训练一个简单的 RL 模型 (CartPole + PPO)

这个脚本演示了在 Docker 环境中训练 RL 模型的基本流程。
运行后会生成 ppo_cartpole.zip 模型文件。
"""

from stable_baselines3 import PPO
import gymnasium as gym


def main():
    print("=" * 50)
    print("Starting RL Training: CartPole + PPO")
    print("=" * 50)
    
    # 1. 创建环境
    print("\n[1/4] Creating environment...")
    env = gym.make("CartPole-v1")
    print(f"Environment: CartPole-v1")
    
    # 2. 创建模型
    print("\n[2/4] Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 3. 训练 (10步快速测试)
    print("\n[3/4] Training (10 timesteps)...")
    model.learn(total_timesteps=10)
    
    # 4. 保存模型
    print("\n[4/4] Saving model...")
    model.save("ppo_cartpole")
    print(f"Model saved to: ppo_cartpole.zip")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

