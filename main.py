from PPO import PPO
from net import NeuralNet


def main(env):
    agents = {
        z:PPO(
            NeuralNet,
        54,
        5,
            z
        )
        for z in env.possible_agents
    }

    i = 1

    while i < 1000:
        env.reset()
        for agent in env.agent_iter():
           agents[agent].act_masked(env)

        for agent in agents.values():
            agent.end_of_game()

        if i%16 == 0:
            for agent in agents.values():
               agent.learn_step_masked()
        i+= 1
