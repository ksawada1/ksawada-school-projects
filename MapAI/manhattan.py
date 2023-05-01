import random
import math
from copy import deepcopy
import time
from statistics import mean

"""
MAP example = 
[[1. 0. 0. 2. 0. 2.]
 [0. 0. 0. 2. 0. 0.]
 [2. 0. 0. 0. 2. 0.]
 [0. 0. 0. 2. 0. 2.]
 [0. 0. 2. 0. 0. 0.]
 [2. 0. 0. 0. 2. 3.]]

 1: START
 2: Edges
 3: GOAL

 ↑こういうマップをランダムにgenerateする。

"""
class Node:
    def __init__(self, direction=None, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.dir = direction

        self.g = 0
        self.h = 0 
        self.f = 0
    
    def __eq__(self, other):
        return self.position == other.position

class Map:
    def __init__(self, width):
        self.width = width
        self.map_size = width * width
        self.path = 0
        self.map = self.map_generator()
        self.current_x = 0
        self.current_y = 0
        self.actions = []
        self.update_actions()
        self.done = False
        self.moves = 0
        

    def map_generator(self):
        # available_route:0, start: 1, goal: 3, obstacles: 2

        map = [[0 for _ in range(self.width)] for _ in range(self.width) ]
        map[0][0] = 1   #START
        map[-1][-1] = 3  #GOAL
        
        for i in range(self.width-1):    
            h = random.randrange(self.width)
            w = random.randrange(self.width)
            
            while map[h][w] != 0:
                h = random.randrange(self.width)
                w = random.randrange(self.width)
            map[h][w] = 2
        return map


    def calculate_distance_to_goal(self):
        a = self.current_y - (self.width -1)
        b = self.current_x - (self.width - 1)
        return math.sqrt(a * a + b * b)

    
    def calculate_distance_of_path_straight(self):
        return self.moves
        

    def move (self, dir):
        if dir in self.actions:
            self.moves += 1
            if dir == "Up":
                self.current_x -= 1
            if dir == "Left":
                self.current_y -= 1
            if dir == "Down":
                self.current_x += 1
            if dir == "Right":
                self.current_y += 1
            if dir == "Up_Left":
                self.current_x -= 1
                self.current_y -= 1
            if dir == "Up_Right":
                self.current_x -= 1
                self.current_y += 1            
            if dir == "Down_Left":
                self.current_x += 1
                self.current_y -= 1
            if dir == "Down_Right":
                self.current_x += 1
                self.current_y += 1
        self.update_actions()


    def update_actions(self):
        if self.map[self.current_x][self.current_y] == 3:
            self.done = True

        self.actions = []
        if (self.current_x != 0) and (self.map[self.current_x-1][self.current_y]) != 2:
            self.actions.append("Up")
        if (self.current_x != 0) and (self.current_y != 0) and (self.map[self.current_x-1][self.current_y-1]) != 2:
            self.actions.append("Up_Left")
        if (self.current_y != 0) and (self.map[self.current_x][self.current_y-1])!= 2:
            self.actions.append("Left")     
        if (self.current_x != self.width-1) and (self.current_y != 0) and (self.map[self.current_x+1][self.current_y-1]) != 2:
            self.actions.append("Down_Left")   
        if (self.current_x != self.width-1) and (self.map[self.current_x+1][self.current_y]) != 2:
            self.actions.append("Down")   
        if (self.current_x != self.width-1) and (self.current_y != self.width-1) and (self.map[self.current_x+1][self.current_y+1]) != 2:
            self.actions.append("Down_Right")                     
        if (self.current_y != self.width-1) and (self.map[self.current_x][self.current_y+1])!= 2:
            self.actions.append("Right")     
        if (self.current_x != 0) and (self.current_y != self.width-1) and (self.map[self.current_x-1][self.current_y+1]) != 2:
            self.actions.append("Up_Right") 


    def display(self):
        disp = deepcopy(self.map)
        disp[self.current_x][self.current_y] = 4
        for r in disp:
            print(r)
        print("------------------------------------")


################################# モデル ########################################
class Model:
    def __init__(self):
        self.map = Map(250)

    def get_actions(self):
        return self.map.actions    

    def apply_action(self, action):
        self.map.move(action)
    
    def done(self):
        return self.map.done

    def print_board(self):
        self.map.display()


############################## 環境モデル ##################################### 
class Environment:
    def __init__(self):
        self.model = Model()
    
    def get_env(self):
        return self.model.map
    
    def apply_action(self, action):
        self.model.apply_action(action)
        
    def done(self):
        return self.model.done()

    def show_state(self):
        self.model.map.display()
    

############################## エージェント　モデル ##################################### 
class Agent:
    def __init__(self):
        self.model = Model()
    
    def update_from_env(self, env):
        self.model.map = env
    
    def agent_function(self, env):
        self.update_from_env(env)
        # actions = self.model.get_actions()
        # action = random.choice(actions)
        return self.a_star(self.model.map, (self.model.map.current_x, self.model.map.current_y), (self.model.map.width-1, self.model.map.width-1))[1]

    def evaluate(self):
        return self.model.map.calculate_distance_to_goal()

    def a_star(self, map, start, goal):
        start_node = Node(None, None, start)   
        start_node.g = start_node.h = start_node.f = 0
        goal_node = Node(None, None, goal)
        goal_node.g = goal_node.h = goal_node.f = 0

        open_lst = []
        closed_lst = []

        #start node
        open_lst.append(start_node)

        while len(open_lst) > 0:
            curr_pos = open_lst[0]
            curr_i = 0
            for i, j in enumerate(open_lst):
                if j.f < curr_pos.f:
                    curr_pos = j
                    curr_i = i
            open_lst.pop(curr_i)
            closed_lst.append(curr_pos)

            if curr_pos == goal_node:
                path = []
                curr = curr_pos
                while curr:
                    path.append(curr.dir)
                    curr = curr.parent
                return path[::-1] 
            
            child = []
            for new_pos in [('Left',(0,-1)),('Right',(0,1)),('Up',(-1,0)),('Down',(1,0)),('Up_Left',(-1,-1)),('Up_Right',(-1,1)),('Down_Left',(1,-1)),('Down_Right',(1,1))]:
                n_pos = (curr_pos.position[0] + new_pos[1][0], curr_pos.position[1]+ new_pos[1][1])
                if n_pos[0]>(len(map.map)-1) or n_pos[0]<0 or n_pos[1]>(len(map.map[len(map.map)-1])-1) or n_pos[1]<0:
                    continue
                #you cannot step on edge(2)
                if map.map[n_pos[0]][n_pos[1]] == 2:
                    continue
                new = Node(new_pos[0], curr_pos, n_pos)
                child.append(new)
            
            for c in child:
                good = True
                for closed_c in closed_lst:
                    if c == closed_c:
                        good = False
                        break
                c.g = curr_pos.g + 1
                c.h = (goal_node.position[0] - c.position[0]) + (goal_node.position[1] - c.position[1])
                c.f = c.g + c.h

                for o in open_lst:
                    if c == o and c.g > o.g:
                        good = False
                        break
                if good:
                    open_lst.append(c)
        print('no path')


############################## MAIN ##################################### 
def main():
    time_result = []
    dis_result = []
    pm = []
    Low, High = 0, 100


    for k in range(Low,High):
        start_time = time.time()
        env = Environment()     
        agent = Agent()     
        while not env.done():
            print("distance to goal: ", env.model.map.calculate_distance_to_goal() )
            env.show_state()     
            percepts = env.get_env()
            action = agent.agent_function(percepts)        
            env.apply_action(action)
        env.show_state()  

        dis_to_goal = env.model.map.calculate_distance_to_goal()
        total_dis = env.model.map.calculate_distance_of_path_straight()
        print("distance to goal : ", dis_to_goal)
        print("total distance of the path: ", total_dis )
        dis_result.append(total_dis)

        t = (time.time() - start_time)
        time_result.append(t)
        print("--- %s seconds ---" % t )
        pm.append((-total_dis)+(-10*t))

    print("how many times run : ", len(time_result))
    print("Average Time : ", (round(mean(time_result),4)))
    print("Average Distance : ",  (round(mean(dis_result),4)))
    print("Average Performance Measure : ",  (round(mean(pm),4)))
    print("MIN of PM : ", (round(min(pm),4)))
    print("MAX of PM : ", (round(max(pm),4)))

main()



