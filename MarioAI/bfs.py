from argparse import Action
import time
import random
import queue
from copy import deepcopy


#マリオs
class Mario:
    def __init__(self):
        self.mario = "Mario"
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0
        self.life = 1
    
    def jump(self):
        self.dy = 1
    
    def move_right(self):
        self.dx = 1
    
    def move_left(self):
        self.dx = -1



#マッシュルーム
class Gumma:
    def __init__(self):
        self.gumma = "Gumma"
        self.y = 0

# 城＝ゴール
class Castle:
    def __init__(self):
        self.castle = "Castle"
        self.x = 10


# xをランダムに動く
# def random_move_left_x(self):
#     self.x += random.randint(0,5)
#     return self.x


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Model:
    def __init__(self):
        self.mario = Mario()
        self.gummas = []
        self.castle = Castle()
        # self.max_time_steps = 100
        # self.time_steps = 0
        self.action = ['right','left', 'jump', 'wait']
        self.win = False

    def get_action_list(self):
        if self.mario.life > 0:
            return self.action
        else: 
            return []
    
    #リーガルアクション　リスト
    def apply_action(self, action):
        if action == "right":
            self.mario.move_right()
        elif action == "left":
            self.mario.move_left()
        elif action == "jump":
            self.mario.jump()   
        elif action == "wait":
            pass
                  
    
    #state all the things mario can see
    # def get_state(self):????????
    #     return self.mario, self.gummas, self.castle

    def done(self):
        #ゴール
        if self.mario.x >= self.castle.x:
            self.win = True
            return True
        #DIE
        elif self.mario.life <= 0:
            return True
        return False            #ゲームが続く

    def update_time_Step(self):
        #move mario & gumma small amount
        if self.mario.life>0:
            for num, gumma in enumerate(self.gummas):
                if gumma.x == self.mario.x:
                    if self.mario.y == 1:
                        self.gummas.pop(num)
                    else:
                        self.mario.life -= 1
            for gumma in self.gummas:
                gumma.x -= 1

            self.mario.y = 0
            self.mario.x += self.mario.dx
            if self.mario.x<0:
                self.mario.x = 0
            self.mario.y += self.mario.dy
            self.mario.dx = 0
            self.mario.dy = 0      

    def add_gumma(self, Xposition):
        newGumma = Gumma()
        newGumma.x = Xposition
        self.gummas.append(newGumma)


    def goal_test(self):
        if self.done():
            return self.win
        return False

    def copy(self): #deep copy in python 
        return deepcopy(self)

#serching solution
    def result(self,a): #conbination of copy and applyAction
        copied = self.copy()
        copied.apply_action(a)
        copied.update_time_Step()
        return copied


#環境モデル----------------------------------------------------------------------------------------------------------------------------------------------------
class Environment:
    def __init__(self):
        self.model = Model()
        self.model.add_gumma(4)
        self.model.add_gumma(3)
   
    def get_percept_list(self): # what info is important to make choice, ask env to match                   
        #put env observation in string
        #mario's position, speed, if he is in air or in ground
        #gumma's position
        percept={"mario":self.model.mario,"gummas": self.model.gummas}
        return percept
        
    def apply_action(self, action):
        self.model.apply_action(action)

    def done(self):
        return self.model.done()

    def show_state(self):
        castle =['           c           ',
                 '  c       ccc       c  ',
                 ' ccc     ccccc     ccc ',
                 'cc cc c  cc cc  c cc cc',
                 'ccccccccccccccccccccccc',
                 'cccc  ccccccccccc  cccc',
                 'cccc  ccccccccccc  cccc',
                 'cccccccccc   cccccccccc',
                 'cccccccccc   cccccccccc']

        print("\n\n\n\n")
        for gumma in self.model.gummas:
            print(gumma.x)
        print("mario x:",self.model.mario.x)
        print("mario y:",self.model.mario.y)

        for i in range(len(castle)):
            y = (len(castle)-1)-i
            for j in range(self.model.castle.x):
                printed = False
                if self.model.mario.y == y and self.model.mario.x == j:
                    print("M", end = "")
                    printed = True
                else:
                    if y == 0:
                        for gumma in self.model.gummas:
                            if gumma.x == j:
                                print("G", end = "")
                                printed = True
                if not printed:
                    print(" ", end = "")
            print(castle[i])
        for i in range(len(castle[0])+self.model.castle.x):
            print("#", end = "")
        print()
    
    def time_step_update(self):
        self.model.update_time_Step()

#エージェントモデル-------------------------------------------------------------------------------------------------------------------------------------------------
class Agent:
    def __init__(self):
        self.model = Model()
        self.action = []
        self.done = False
    
    def update_from_percept_list(self, percept_lst):                          
        #update data member to match the percepts, ask agent to match
        self.model.mario = percept_lst["mario"]
        self.model.gummas = percept_lst["gummas"]

    def agent_function(self, percepts):    
        self.update_from_percept_list(percepts)
        action = self.bfs()
        return action

    def bfs(self):
        frontier = queue.Queue()
        frontier.put(("none",self.model.copy()))
        while not frontier.empty():
            node = frontier.get()
            if node[1].goal_test():
                return node[0]
            for a in node[1].get_action_list():
                act = node[0]
                if act == "none":
                    act = a
                s_child = node[1].result(a)
                frontier.put((act,s_child))

    def random_agent(self):
        actions = self.model.get_action_list()
        action = random.choice(actions)
        return action

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    env = Environment()     #assumes env is randomly populated among possible anv
    agent = Agent()       #assumers random agent class exists

    while not env.done():
        env.show_state()     # show all the states such as mario, mashroom
        percepts = env.get_percept_list()
        action = agent.agent_function(percepts)        #random = agent_function
        print(action)
        env.apply_action(action)
        env.time_step_update()
        
        #display_performance_measure(env, agent)

        #return

        # s1 = s.copy()
        # s1.applyAction(a)

    # def updateMoveMushroom():
    #     #move mushroom mushroom doesnt jump
    #     pass


    # def getPerceptList():
    #     #put env observation in string
    #     percept=[]
    #     return percept

    # def updateFromPerceptList(percept_lst):
    #     pass
main()