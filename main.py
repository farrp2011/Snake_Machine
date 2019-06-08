#my suff
from snakeObj import *
#from policy_gradients_torch import DeepQNetwork
#from policy_gradients_torch import Agent
from policy_gradients_torch import Agent
from utils import plotLearning

#Libary stuff
from graphics import *
from datetime import datetime
import csv
import os

#import gym
import numpy as np
import matplotlib.pyplot as plt


csv_output = []
csv_output.append(["Round", "Total Reward", "Score", "Steps"])


def main():

    score_history = []
    n_episodes = 100000
    score = 0
    renderInterval = 0
    renderGame = 100 #we will render every X number of games


    #__init__( lr, input_dims, gamma=0.99, n_actions=4, lt_size=256, l2_size=256:
    agent = Agent(lr=0.0005, input_dims=[404], gamma=0.99, n_actions=4, lt_size=64, l2_size=64, l3_size=64)
    win = None
    game_score = 0

    gameNotOver = True
    user_quit = False
    #while user_quit == False:
    for i in range(n_episodes):
        gameNotOver = True

        if(renderInterval == renderGame):
            win = GraphWin("Snake Game", WIDTH(), HEIGHT())
            game = GameObj(win)
        else:
            game = GameObj()
            #here we will render the X game

        score = 0

        observation, reward, done, info = game.getObservatoin()

        while(gameNotOver == True):
            #we are moving through each step of the game here
            action = agent.choose_action(observation)
            gameNotOver = game.logic(action)
            #print(action)
            observation_, reward, done, info = game.getObservatoin()

            game_score = game.scoreNum

            agent.store_rewards(reward)
            observation = observation_
            score += reward

        if(renderInterval == renderGame):
            #we need to make sure to close the window and reset the interval
            win.close()
            renderInterval = 0

        renderInterval += 1
        score_history.append(score)
        csv_output.append([i, score, game.scoreNum, game.steps])

        os.system("clear")
        print("Game  : ", i)
        print("Reward: ", score)
        print("Score : ", game.scoreNum)
        print("Steps : ", game.steps)


        agent.learn()
        n_episodes -= 1
    agent.saveModel()


main()

now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H:%M:%S")

with open("Snake_Game"+date_time+".csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for lines in csv_output:
        spamwriter.writerow(lines)
    csvfile.close()
