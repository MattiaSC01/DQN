import gymnasium as gym


def play_random_game(env):
    done = False
    observation, info = env.reset()
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


print("Which game do you want to play?")
print("Here is a list of available games:")
all_games = gym.envs.registry.keys()
for i, game in enumerate(all_games):
    print(f"{i}. {game}")
game = input()
while game not in all_games:
    if game.isnumeric() and 0 <= int(game) < len(all_games):
        game = list(all_games)[int(game)]
        break
    print("Game not found, please try again.")
    game = input()

# print("Do you want to render the environment? (y/n)")
# render = input().lower() in ["yes", "y"]
render = True

if render:
    env = gym.make(game, render_mode="human")
else:
    env = gym.make(game)

print("Action space:")
print(env.action_space)
print("Observation space:")
print(env.observation_space)
print("Reward range:")
print(env.reward_range)
print("Metadata:")
print(env.metadata)

if render:
    play_random_game(env)

env.close()
