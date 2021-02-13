#my suff
from snakeObj import *
from simple_dqn_tf import DeepQNetwork, Agent
#from utils import plotLearning

#Libary stuff
from graphics import *
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import os
import numpy as np



csv_output = []
csv_output.append(["Round Number", "Total Reward", "Score", "Steps"])

def main():
    
    lr = 0.0002
    n_episodes = int(input("How many times?"))
    n_episodes += 50
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=lr, input_dims=[404], n_actions=4, mem_size=2000000, n_games=n_episodes, batch_size=128)

    scores = []
    eps_history = []
    
    
    score = 0
    renderInterval = 499
    renderGame = 500 #we will render every X number of games

    win = None
    game_score = 0

    gameNotOver = True
    user_quit = False
    #while user_quit == False:
    for i in range(n_episodes):
        gameNotOver = True
        game = None
        if(i > (n_episodes -50)):
            junkVar = input("Press Enter To Contunue...")
            win = GraphWin("Snake Game", WIDTH(), HEIGHT())
            game = GameObj(win)
        else:
            game = GameObj()
            #here we will not render the game

        score = 0

        observation, reward, done, info = game.getObservatoin()#this is our reset 
        observation = np.array(observation)
        while(gameNotOver == True):
            #we are moving through each step of the game here
            action = agent.choose_action(observation)
            
            gameNotOver = game.logic(action)
            #print(action)
            observation_, reward, done, info = game.getObservatoin()
            
            game_score = game.scoreNum

            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = np.array(observation_)
            agent.learn()

        if(i > (n_episodes - 50)):
            #we need to make sure to close the window and reset the interval
            win.close()
            renderInterval = 0

        eps_history.append(agent.epsilon)
        renderInterval += 1
        csv_output.append([i, score, game.scoreNum, game.steps])

        os.system("clear")
        print("Game  : ", i)
        print("Reward: ", score)
        print("Score : ", game.scoreNum)
        print("Steps : ", game.steps)


        
        n_episodes -= 1


main()

now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H:%M:%S")

with open("Snake_Game"+date_time+".csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for lines in csv_output:
        spamwriter.writerow(lines)
    csvfile.close()
