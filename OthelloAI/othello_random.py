import random
from copy import deepcopy 

# Othello - 4 x 4 board version


###########################　ボックス　#####################################
class Box:
    def __init__(self, empty):
        self.empty = empty
        self.marker = empty

    def empty_box(self): 
        return self.marker == self.empty

    def set_marker(self, mark): 
        self.marker = mark

    def get_marker(self): 
        return self.marker


###########################　オセロ　##################################### 
class Othello:
    def __init__(self):
        self.empty = "+"
        self.players = ('●','◯')
        self.done = False
        self.turn = self.players[0]
        self.actions = []
        self.height = 4
        self.width = 4
        self.board = [Box(self.empty) for _ in range(self.height*self.width)]
        self.board[self.height//2*self.width+self.width // 2].set_marker(self.players[0])
        self.board[self.height//2*self.width+self.width // 2-1].set_marker(self.players[1])
        self.board[self.height//2*self.width+self.width // 2-self.width].set_marker(self.players[1])
        self.board[self.height//2*self.width+self.width // 2-1-self.width].set_marker(self.players[0])
        self.update_actions()


    def score(self):
        score = {self.players[0]:0,self.players[1]:0}
        for box in self.board:
            if box.get_marker() == self.players[1]:
                score[self.players[1]] +=1
            if box.get_marker() == self.players[0]:
                score[self.players[0]] +=1
        return score


    def move(self, pos):
        if pos in self.actions:
            for i in self.reverse(pos,self.turn):
                self.board[i].set_marker(self.turn)
            self.board[pos].set_marker(self.turn)
            self.change_turn()
            self.update_actions()
            if not self.actions:
                self.change_turn()
                self.update_actions()
            if not self.actions:
                self.done = True
        else:
            print("cant put a", self.turn, pos)


    def change_turn(self):
        if self.turn == self.players[0]:
            self.turn = self.players[1]
        else:
            self.turn = self.players[0]


    def reverse(self, pos, mark):
        flips = []
    #right
        newF = []
        cur = pos + 1
        while cur%self.width != 0:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur += 1
    #left
        newF = []
        cur = pos-1
        while cur%self.width != self.width-1:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur -= 1
    #down
        newF = []
        cur = pos+self.width
        while cur < len(self.board):
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur += self.width
    #up
        newF = []
        cur = pos-self.width
        while cur >= 0:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur -= self.width
    #right-down
        newF = []
        cur = pos+self.width+1
        while cur < len(self.board) and cur%self.width != 0:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur += self.width+1
    #left-down
        newF = []
        cur = pos+self.width-1
        while cur < len(self.board) and cur%self.width != self.width-1:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur += self.width-1
    #right-up
        newF = []
        cur = pos-self.width+1
        while cur >= 0 and cur%self.width != 0:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur -= self.width+1
    #left-up
        newF = []
        cur = pos-self.width-1
        while cur >= 0 and cur%self.width != self.width-1:
            if self.board[cur].get_marker() == mark:
                flips += newF
                break
            if self.board[cur].get_marker() == self.empty:
                break
            newF.append(cur)
            cur -= self.width - 1
    #return it 
        return flips


    def update_actions(self):
        actions = []
        for i in range(len(self.board)):
            if self.board[i].empty_box():
                if self.reverse(i,self.turn):
                    actions.append(i)
        self.actions = actions


    def display(self):
        edge = ""
        for i in range(self.height + 2):
            edge += "-"
        print(edge)
        for i in range(self.height):
            line = "|"
            for j in range(self.width):
                line += self.board[i * self.width + j].get_marker()
            line += "|"
            print(line)
        print(edge)


############################## モデル ########################################
class Model:
    def __init__(self):
        self.othello = Othello()

    def get_actioins(self):
        return self.othello.actions    

    def apply_action(self, action):
        self.othello.move(action)
        self.check_who_won()
    
    def done(self):
        return self.othello.done

    # result ???
    def check_who_won(self):
        if self.othello.done:
            if (self.othello.score()[self.othello.players[0]]) > (self.othello.score()[self.othello.players[1]]):
                print("Result:  BLACK ● WON!!")
            else:
                print("Result:  WHITE ◯ WON!!")

    def print_board(self):
        self.othello.display()


############################## 環境モデル ##################################### 
class Environment:
    def __init__(self):
        self.model = Model()
    
    def get_env(self):
        return self.model.othello
    
    def apply_action(self, action):
        self.model.apply_action(action)
        
    def done(self):
        return self.model.done()

    def show_state(self):
        self.model.othello.display()
    

############################## エージェント　モデル ##################################### 
class Agent:
    def __init__(self):
        self.model = Model()
    
    def update_from_env(self, env):
        self.model.othello = env
    
    def agent_function(self, env):
        self.update_from_env(env)
        actions = self.model.get_actioins()
        action = random.choice(actions)

        if self.model.othello.players[0]:
            print("BLACK ● action: ", action)
        else:
            print("WHITE ◯ action: ", action)
        return action
        


############################## MAIN ##################################### 
def main():
    env = Environment()     
    agent = Agent()     

    while not env.done():
        print("current score: ", env.model.othello.score() )
        env.show_state()     
        print(env.model.othello.turn,"available actions: ", env.model.othello.actions)
        percepts = env.get_env()
        action = agent.agent_function(percepts)        
        env.apply_action(action)
        
main()

# othello = Othello()
# while not othello.done:
#     print("score: ",othello.score())
#     othello.display()
#     print(othello.turn,"choose from actions_list: ",othello.actions)
#     othello.move(int(input()))
# print("\n\nFinal score: ",othello.score())
# othello.display()

