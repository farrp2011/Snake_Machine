from graphics import *
import random
import time
import os


def ARROW_UP():
    #return ("w")
    return(0)


def ARROW_DOWN():
    #return ("s")
    return (1)

def ARROW_LEFT():
    #return ("a")
    return (2)

def ARROW_RIGHT():
    #return ("d")
    return (3)

def GAME_SPEED():
    return (.2)  # the small the numer the faster the game


def BACKGROUND_COLOR():
    return (color_rgb(10, 207, 0))


def PLAYER_COLOR():
    return (color_rgb(0, 155, 149))


def LINE_COLOR():
    return (color_rgb(0, 0, 0))


def FOOD_COLOR():
    return (color_rgb(253, 0, 6))


def HUB_COLOR():
    return (color_rgb(255, 113, 0))


def WHITE():
    return (color_rgb(255, 255, 255))


def BOX_NUM():
    return (20)  # number of boxs in the x and y directions


def FULL_BOX_NUM():
    return (BOX_NUM() * BOX_NUM())


def PLAY_AREA():
    return (500)  # size of the play area


def HUB_AREA():
    return (int(50))  # Height of the play area


def WIDTH():
    return (PLAY_AREA())  # width of the play area


def HEIGHT():
    return (int(PLAY_AREA() + HUB_AREA()))  # toatal Height of the screen


def BOX_SIZE():
    return (PLAY_AREA() / BOX_NUM())  # size of boxes


class GameObj():
    """This is the Snake Game Object
	 We will have 4 main parts
	 1. The contructor that set everything up
	 2. The Display method that displays stuff
	 3. Game logic stuff that runs the game for one cycle
	 4. some helper methods
		4.1 delay method that delays the game till it is ready to for the game logic
		4.2 spawfood method that find a place to put food
		4.3 onkeydown event method to know when the user pressed something.(Might be a little hard...)
	 """

    def spawnFood(self):
        newX = 0;
        newY = 0;

        goodPoint = False

        while (goodPoint == False):
            goodPoint = True
            newX = random.randint(0, BOX_NUM() - 1);
            newY = random.randint(0, BOX_NUM() - 1);
            for part in self.body:
                if (newX == part["x"] and newX == part["y"]):
                    goodPoint = False
                    break
        self.food["x"] = newX
        self.food["y"] = newY

    def isGraphWin(self, canvas):
        if type(canvas) is GraphWin:
            print("")
        else:
            raise Exception('First Parameter Must be GraphWin Object')

    def display(self):

        if (self.isDisplay == True):
            return  # we dont't what to run this twice

        self.gameMessage = Text(Point(round(PLAY_AREA() / 2), round(PLAY_AREA() / 2)), "Use W,A,S,D Keys To Start")
        self.gameMessage.setSize(26)
        self.gameMessage.setFace("arial")
        self.gameMessage.setTextColor("white")

        # background
        self.backGround = Rectangle(Point(0, 0), Point(WIDTH(), HEIGHT()))
        self.backGround.setFill(BACKGROUND_COLOR())

        # hub
        self.hub = Rectangle(Point(0, PLAY_AREA()), Point(WIDTH(), HEIGHT()))
        self.hub.setFill(HUB_COLOR())
        self.hub.setOutline(HUB_COLOR())

        # food
        self.foodBox = Rectangle(Point(self.food["x"] * BOX_SIZE(), self.food["y"] * BOX_SIZE()),
                                 Point((1 + self.food["x"]) * BOX_SIZE(), (1 + self.food["y"]) * BOX_SIZE()))
        self.foodBox.setFill(FOOD_COLOR())
        self.foodBox.setOutline(FOOD_COLOR())

        self.backGround.draw(self.canvas)
        self.hub.draw(self.canvas)
        self.foodBox.draw(self.canvas)

        # Here we are making the player.
        # it is kinda annoying because I have to make every
        self.bodyPart = [None] * FULL_BOX_NUM()
        for i in range(FULL_BOX_NUM()):
            self.bodyPart[i] = Rectangle(Point(-1 * BOX_SIZE(), -1 * BOX_SIZE()),
                                         Point((1 + -1) * BOX_SIZE(), (1 + -1) * BOX_SIZE()))

            self.bodyPart[i].setFill(PLAYER_COLOR())
            self.bodyPart[i].setOutline(PLAYER_COLOR())
            self.bodyPart[i].draw(self.canvas)

        # the Eye (this is so the player can see where the head is)
        eyeCenX = self.getEyeX()
        eyeCenY = self.getEyeY()
        pupilCenX = self.getPupilX()
        pupilCenY = self.getPupilY()

        # the white of the eye
        self.eye = Circle(Point(eyeCenX, eyeCenY), round(BOX_SIZE() / 4))
        self.eye.setOutline(WHITE())
        self.eye.setFill(WHITE())
        self.eye.draw(self.canvas)
        # the little black pupil
        self.pupil = Circle(Point(pupilCenX, pupilCenY), round(BOX_SIZE() / 8))
        self.pupil.setOutline(LINE_COLOR())
        self.pupil.setFill(LINE_COLOR())
        self.pupil.draw(self.canvas)

        # now we draw the grid
        for i in range(BOX_NUM() + 1):
            line1 = Line(Point(0, i * BOX_SIZE()), Point(PLAY_AREA(), i * BOX_SIZE()))
            line2 = Line(Point(i * BOX_SIZE(), 0), Point(i * BOX_SIZE(), PLAY_AREA()))
            line1.draw(self.canvas)
            line2.draw(self.canvas)

        self.score = Text(Point(round(PLAY_AREA() / 2), round(PLAY_AREA() + HUB_AREA() / 2)),
                          "Score: " + str(self.scoreNum))
        self.score.setSize(26)
        self.score.setFace("arial")
        self.score.setTextColor("white")
        self.score.draw(self.canvas)
        self.gameMessage.draw(self.canvas)
        update(30)  # this function forces everything to be drawn

    def getEyeX(self):
        eyeCenX = None
        if (self.direction == ARROW_UP()):
            eyeCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == ARROW_DOWN()):
            eyeCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == ARROW_RIGHT()):
            eyeCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() * .75)
        elif (self.direction == ARROW_LEFT()):
            eyeCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 4)
        elif (self.direction == None):
            eyeCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        # print("Eye x="+str(eyeCenX))
        return (eyeCenX)

    def getEyeY(self):
        eyeCenY = None
        if (self.direction == ARROW_UP()):
            eyeCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 4)
        elif (self.direction == ARROW_DOWN()):
            eyeCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() * .75)
        elif (self.direction == ARROW_RIGHT()):
            eyeCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == ARROW_LEFT()):
            eyeCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == None):
            eyeCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        # print("Eye y="+str(eyeCenY))
        return (eyeCenY)

    def getPupilX(self):
        pupilCenX = None
        if (self.direction == ARROW_UP()):
            pupilCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == ARROW_DOWN()):
            pupilCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == ARROW_RIGHT()):
            pupilCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() * .875)
        elif (self.direction == ARROW_LEFT()):
            pupilCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 8)
        elif (self.direction == None):
            pupilCenX = self.body[0]["x"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        return (pupilCenX)

    def getPupilY(self):
        pupilCenY = None
        if (self.direction == ARROW_UP()):
            pupilCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 8)
        elif (self.direction == ARROW_DOWN()):
            pupilCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() * .875)
        elif (self.direction == ARROW_RIGHT()):
            pupilCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == ARROW_LEFT()):
            pupilCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        elif (self.direction == None):
            pupilCenY = self.body[0]["y"] * BOX_SIZE() + round(BOX_SIZE() / 2)
        return (pupilCenY)

    def getNewX(self, anchor, x):
        x = x - anchor.getX()
        return (round(x))

    def getNewY(self, anchor, y):
        y = y - anchor.getY()
        return (round(y))

    def getBodyPosX(self, x):
        return (x * BOX_NUM())

    def getBodyPosY(self, y):
        return (y * BOX_NUM())

    def logic(self, button):

        newBodyPart = None

        if (self.addToBody == True):
            self.addToBody == False
            # newBodyPart = {x:gameObj.body[gameObj.body.length-1].x,y:gameObj.body[gameObj.body.length-1].y};

            newBodyPart = {"x": self.body[self.length]["x"], "y": self.body[self.length]["y"]}

        # this adds the new body part
        if (newBodyPart != None):
            self.addToBody = False
            self.length = self.length + 1
            self.body.append({"x": newBodyPart["x"], "y": newBodyPart["y"]})

        # here we will move each body part but the head

        i = len(self.body) - 1
        while (i != 0):
            self.body[i]["x"] = self.body[i - 1]["x"]
            self.body[i]["y"] = self.body[i - 1]["y"]
            i = i - 1

        # set Direction
        if (button == ARROW_UP() and self.lastDir != ARROW_DOWN()):
            self.direction = ARROW_UP()
        elif (button == ARROW_DOWN() and self.lastDir != ARROW_UP()):
            self.direction = ARROW_DOWN()
        elif (button == ARROW_LEFT() and self.lastDir != ARROW_RIGHT()):
            self.direction = ARROW_LEFT()
        elif (button == ARROW_RIGHT() and self.lastDir != ARROW_LEFT()):
            self.direction = ARROW_RIGHT()

        # This is the Direction we are going
        if (self.direction == ARROW_UP()):
            self.body[0]["y"] = self.body[0]["y"] - 1
        elif (self.direction == ARROW_DOWN()):
            self.body[0]["y"] = self.body[0]["y"] + 1
        elif (self.direction == ARROW_LEFT()):
            self.body[0]["x"] = self.body[0]["x"] - 1
        elif (self.direction == ARROW_RIGHT()):
            self.body[0]["x"] = self.body[0]["x"] + 1

        # did the player find food?
        if (self.body[0]["x"] == self.food["x"] and self.body[0]["y"] == self.food["y"]):
            self.addToBody = True
            self.spawnFood()
            self.scoreNum = self.scoreNum + 1

        # did the player lose?
        # did the play fall off the map?
        if (self.body[0]["x"] < 0 or self.body[0]["y"] < 0 or self.body[0]["x"] > BOX_NUM() - 1 or self.body[0]["y"] > BOX_NUM() - 1):
            # here the player fell off the map
            self.gameOver = True
            self.updateDisplay()
            return (False)

        #os.system("clear")
        #print("Head Location X:" + str(self.body[0]["x"]) + " Y:" + str(self.body[0]["y"]))
        #print("2nd part Location X:" + str(self.body[1]["x"]) + " Y:" + str(self.body[1]["y"]))
        #print("Food Location X:" + str(self.food["x"]) + " Y:" + str(self.food["y"]))
        #print("Body Len:" + str(len(self.body)))

        num = len(self.body)
        for i in range(1, num):

            if (self.body[0]["x"] == self.body[i]["x"] and self.body[0]["y"] == self.body[i]["y"] and self.direction != None):
                # here the player ran into themslef
                self.gameOver = True
                self.updateDisplay()
                return (False)

        self.updateDisplay()
        return (True)

    def updateDisplay(self):
        if (self.gameOver == True):
            self.gameMessage.setText("Game Over")
            return ()
        elif (self.direction == None):
            self.gameMessage.setText("Use W,A,S,D Keys To Start")
        else:
            self.gameMessage.setText("")
        # food
        self.foodBox.move(self.getNewX(self.foodBox.getP1(), self.food["x"] * BOX_SIZE()),
                          self.getNewY(self.foodBox.getP1(), self.food["y"] * BOX_SIZE()))

        # Here we are making the player.
        # it is kinda annoying because I have to make every

        for i in range(len(self.body)):
            self.bodyPart[i].move(self.getNewX(self.bodyPart[i].getP1(), self.body[i]["x"] * BOX_SIZE()),
                                  self.getNewY(self.bodyPart[i].getP1(), self.body[i]["y"] * BOX_SIZE()))

        # the Eye (this is so the player can see where the head is)
        eyeCenX = self.getEyeX()
        eyeCenY = self.getEyeY()
        pupilCenX = self.getPupilX()
        pupilCenY = self.getPupilY()

        # move Pupil and eye
        self.eye.move(self.getNewX(self.eye.getCenter(), eyeCenX), self.getNewY(self.eye.getCenter(), eyeCenY))
        self.pupil.move(self.getNewX(self.pupil.getCenter(), pupilCenX),
                        self.getNewY(self.pupil.getCenter(), pupilCenY))

        # change score
        self.score.setText("Score: " + str(self.scoreNum))
        update(5)  # this function forces everything to be drawn

    def playAgain(self):

        againPointX1 = round(WIDTH() * .25)
        againPointX2 = round(WIDTH() * .75)
        againPointY1 = round(PLAY_AREA() * .25) - 25
        againPointY2 = round(PLAY_AREA() * .25) + 25

        endPointX1 = round(WIDTH() * .25)
        endPointX2 = round(WIDTH() * .75)
        endPointY1 = round(PLAY_AREA() * .75) - 25
        endPointY2 = round(PLAY_AREA() * .75) + 25

        again = Rectangle(Point(againPointX1, againPointY1), Point(againPointX2, againPointY2))
        end = Rectangle(Point(endPointX1, endPointY1), Point(endPointX2, endPointY2))
        again.setFill(HUB_COLOR())
        end.setFill(HUB_COLOR())
        againText = Text(Point(round(PLAY_AREA() / 2), round(PLAY_AREA() * .25)), "Play Again")
        endText = Text(Point(round(PLAY_AREA() / 2), round(PLAY_AREA() * .75)), "Quit")

        again.draw(self.canvas)
        end.draw(self.canvas)
        againText.draw(self.canvas)
        endText.draw(self.canvas)

        selection = False
        madeSelection = False
        while (madeSelection == False):
            anchor = self.canvas.getMouse()
            if (
                    anchor.getX() > againPointX1 and anchor.getY() > againPointY1 and anchor.getX() < againPointX2 and anchor.getY() < againPointY2):
                # are we inside the box of the again box
                madeSelection = True
                # print("They want to play again")
                selection = False
            if (
                    anchor.getX() > endPointX1 and anchor.getY() > endPointY1 and anchor.getX() < endPointX2 and anchor.getY() < endPointY2):
                madeSelection = True
                selection = True
                # print("The Player wanted to end the game")

        again.undraw(self.canvas)
        end.undraw(self.canvas)
        againText.undraw(self.canvas)
        endText.undraw(self.canvas)
        return (selection)

    def getObservatoin(self):
        #We need to flatten everything into a single array and hopefully it all works out
        tempDir = self.direction
        if(tempDir == None):
            tempDir = -1
        outputArr = [-1] * 404 #this is not a 404 error; 400 places and 4 for food(x,y),dirction and score
        #outputArr = [tempDir, self.scoreNum, self.food["x"], self.food["y"]]
        outputArr[0] = tempDir
        outputArr[1] = self.scoreNum
        outputArr[2] = self.food["x"]
        outputArr[3] = self.food["y"]

        reward = 0.0
        i = 0
        while(i < len(self.body)):
            outputArr[i+4] = self.body[i]["x"]
            outputArr[i+4] = self.body[i]["y"]
            i += 1

        #cal reward
        if(self.addToBody == True):
            reward += 25
        else:
            xDis = abs(self.body[0]["x"] - self.food["x"])
            yDis = abs(self.body[0]["y"] - self.food["y"])
            reward = 1/(2*(xDis + yDis))

        #if(self.gameOver == True):
        #    reward = -1
        info = "Head Location X:" + str(self.body[0]["x"]) + " Y:" + str(self.body[0]["y"])
        #reward += self.steps * .001 #we will give a little if it survies

        #if the snake goes in the opposite dirctions no points
        #if(ARROW_UP() == tempDir and self.lastDir == ARROW_DOWN()):
        #    reward = -0.1
        #if(ARROW_DOWN() == tempDir and self.lastDir == ARROW_UP()):
        #    reward = -0.1
        #if(ARROW_LEFT() == tempDir and self.lastDir == ARROW_RIGHT()):
        #    reward = -0.1
        #if(ARROW_RIGHT() == tempDir and self.lastDir == ARROW_LEFT()):
        #    reward = -0.1
        self.steps += 1
        self.lastDir = tempDir
        #observation_, reward, done, info

        if(self.steps > 10000):
            self.gameOver = True


        return(outputArr,reward, self.gameOver, info)

    def __init__(self, canvas):
        self.steps = 0
        self.lastDir = -1
        self.isGraphWin(canvas)
        self.canvas = canvas
        self.gameOver = False
        self.direction = None
        self.scoreNum = 0
        self.body = [{"x": 9, "y": 9}]  # this is array of the body parts
        self.length = 0
        self.addToBody = True
        self.food = {"x": 5, "y": 3}  # this will be reset by the another method soon
        self.spawnFood()
        self.isDisplay = False
        self.display()
        self.updateDisplay()
        self.length = 0


