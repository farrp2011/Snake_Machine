#my stuff
from snakeObj import *
#from policy_gradients_torch import DeepQNetwork
#from policy_gradients_torch import Agent
from policy_gradients_torch import Agent
from utils import plotLearning

#Libary stuff
from graphics import *

#import gym
import numpy as np
import matplotlib.pyplot as plt

def main():


    score_history = []
    n_episodes = 2000
    score = 0
    #__init__( lr, input_dims, gamma=0.99, n_actions=4, lt_size=256, l2_size=256:
    agent = Agent(lr=0.0005, input_dims=[404], gamma=0.99, n_actions=4, lt_size=64, l2_size=64)


    gameNotOver = True
    user_quit = False
    #while user_quit == False:
    for i in range(n_episodes):
        gameNotOver = True
        print("game: ", i ,"Socre: ", score)
        win = GraphWin("Snake Game", WIDTH(), HEIGHT())

        game = GameObj(win)
        observation, reward, done, info = game.getObservatoin()
#        try:
        while(gameNotOver == True):
            action = agent.choose_action(observation)
            gameNotOver = game.logic(action)
            observation_, reward, done, info = game.getObservatoin()
            agent.store_rewards(reward)
            observation = observation_
            score += reward
                #time.sleep(.1)#this is the game speed!

#        except:
#            print("some error but the user probably clicked the exit button")
        #user_quit = game.playAgain() #asks if the user whats to play again but it is not down yet
        win.close()
        score_history.append(score)
        agent.learn()
        n_episodes -= 1
main()

