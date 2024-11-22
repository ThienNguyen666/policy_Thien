import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy, GeneticPolicy

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 1

if __name__ == "__main__":
    # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test GeneticPolicy    
    gen_policy = GeneticPolicy(
        population_size=100,
        generations= 10,
        mutation_rate= 0.2,
        stock_width= 12,
        stock_height= 15
        )
    ep = 0
    while ep < NUM_EPISODES:
        action = gen_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1
env.close()
