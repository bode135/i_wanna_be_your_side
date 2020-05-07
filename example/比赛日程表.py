import numpy as np

length = 6
l_len = length//2


#------------ Init
class Team():
    def __init__(self,index,name):
        self.index = index
        self.name = name

teams = []
for i in range(length):
    team = Team(i,'T'+str(i))
    teams.append(team)
def p_ts():
    global teams
    for i in teams:
        print(i.index,i.name)
#p_ts()

#------------------- Data
def p_teams(l):
    for i in l:
        print(teams[i].index,teams[i].name)
def pp_ts(l1,l2):
    for i, j in dict(zip(l1, l2)).items():
        print(teams[i].name, teams[j].name)


#----------------   algorithm
list1 = list(range(0,l_len))
list2 = list(range(l_len,l_len*2))
games = np.column_stack([list1,list2])
Games = []
for i in range(length-1):
    tmp1 = list1.pop(-1)
    tmp2 = list2.pop(0)
    # tmp1,tmp2
    list1.insert(1,tmp2)
    list2.append(tmp1)
    games = np.column_stack([list1,list2])
    # games
    Games.append(games)
    #pp_ts(games[:,0],games[:,1])
    # print('------',i)


#------------------   print
np.unique(Games)
Games[4]

def print_game(game):
    pp_ts(game[:,0],game[:,1])
def print_igame(igame):
    global Games
    pp_ts(Games[igame][:, 0], Games[igame][:, 1])
for i in range(length-1):
    print('-第 ',i+1,' 天--')
    pp_ts(Games[i][:, 0], Games[i][:, 1])

# for i in range(length-1):
#     print('----- ',i+1)
#     print(Games[i])
#

