import matplotlib
import matplotlib.pyplot as plt #as per the above, much easier to write over and over again
from numpy import random
from matplotlib import colors
from IPython.display import clear_output, display
import pdb
import random
import numpy as np
import os
import getpass

NUMA=1000
NUMB=1000
RAN_A=False
ATTR=10

username=getpass.getuser()
    
EXPERIMENT = 1   
    

if not os.path.exists('/tmp/' + username):
    os.mkdir('/tmp/' + username)
os.system('rm /tmp/' + username + '/*')


# class, defining agents with their position and group membership
class agent:
    def __init__(self,x,y,group):
        self.score = 100 # agent's life score
        self.x = x
        self.y = y
        self.group = group
        
# define a function for creating agents and assigning them to grid
def agentCreator(size,group,groupList,field,n,m):
    # loop through entire group
    for j in range(0,size):
        # select random available location 
        while True:
            # random x coordinate
            x = random.choice(range(0,n))
            # random y coordinate
            y = random.choice(range(0,m))
            # check if spot is available; if not then re-iterate 
            if field[x][y] == None:
                field[x][y] = agent(x=x,y=y,group=group)
                # append agent object reference to group list
                groupList.append(field[x][y])
                # exit while loop; spot on field is taken
                break

# function for creating an initial grid
def initfield(populationSizeA,populationSizeB):
    # initializing new empty grid, using list comprehension in Python
    field_grid = [[None for i in range(0,100)] for j in range(0,100)]
    # create empty list for containing agent references in future, type A & B
    agents_A = []
    agents_B = []
    # assigning random spots to agents of group A and B; 
    agentCreator(size = populationSizeA,
                    group = "A",
                    groupList = agents_A,
                    field = field_grid,
                    n = 100,
                    m = 100)
    agentCreator(size = populationSizeB,
                    group = "B",
                    groupList = agents_B,
                    field = field_grid,
                    n = 100,
                    m = 100)

    # return populated grid and a list of agents in groups A and B
    return field_grid, agents_A, agents_B

# executing above function for a population size of 1000 for both groups
field, agents_A, agents_B = initfield(populationSizeA=NUMA,populationSizeB=NUMB)
# print (np.count_nonzero(np.array(mapfield(field))==2.))

# Now lets plot the 2D space and see what our initial condition looks like
def mapfield(field):
    #.imshow() needs a matrix with float elements;
    population = [[0.0 for i in range(0,100)] for j in range(0,100)]
    # if agent is of type A, put a 1.0, if of type B, pyt a 2.0
    for i in range(0,100):
        for j in range(0,100):
            if field[i][j] == None: # empty
                pass # leave 0.0 in population cell
            elif field[i][j].group == "A": # group A agents
                population[i][j] = 1.0 # 1.0 means "A"
            else: # group B agents
                population[i][j] = 2.0 # 2.0 means "B"
    # return mapped values
    return population

population_grid = mapfield(field)

colormap = colors.ListedColormap(["white","green","blue"])

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(population_grid, cmap=colormap)
plt.title('Agent distribution: Group A green, B blue')
plt.ion()
plt.show()
plt.savefig('/tmp/' + username + '/initial.png')
plt.close()





# function for removing agents from battlefield grid when life score is not strictly positive
def removeAgents(field):
    # identifying agents with life score of score or below - and removing them from the grid
    for i in range(0,len(field)):
        for j in range(0,len(field)):
            if field[i][j]:
                if field[i][j].score <= 0:
                    # remove this agent since life score is not strictly positive
                    field[i][j] = None
        
# function implementing one round of interaction, for an agent of type A
def oneRoundAgentA(i,j,attackRange):
    found_i = None
    found_j = None
    # look in neigbouring cells in same order for each iteration
    list1 = [k for k in range(i-attackRange,i+attackRange+1)]
    list2 = [l for l in range(j-attackRange,j+attackRange+1)]
    if RAN_A:
        random.shuffle(list1)
        random.shuffle(list2)
    for k in list1:
        for l in list2:
            # check for negative index values; if so - break!
            if k < 0 or l < 0:
                break
                # check for index values above 99, if so break!
            if k > 99 or l > 99:
                break
            if field[k][l]:
                if field[k][l].group == "B": # then this is an enemy
                    if found_i == None:
                        found_i = k
                        found_j = l
                    
    # deal damage to identified specie
    if found_i != None:
        field[found_i][found_j].score = field[found_i][found_j].score - random.randint(10,60)
        
# function implementing one round of interaction, for an agent of type B
def oneRoundAgentB(i,j,attackRange):
    found_i = None
    found_j = None
    # look in neigbouring cells in same order for each iteration
    
    # In B's case we look randomly through the surrounding space
    list1 = [k for k in range(i-attackRange,i+attackRange+1)]
    list2 = [l for l in range(j-attackRange,j+attackRange+1)]
    random.shuffle(list1)
    random.shuffle(list2)
    for k in list1:
        for l in list2:
            # check for negative index values; if so - break!
            if k < 0 or l < 0:
                break
                # check for index values above 99, if so break!
            if k > 99 or l > 99:
                break
            if field[k][l]:
                if field[k][l].group == "A": # then this is an enemy
                    if found_i == None:
                        found_i = k
                        found_j = l
    # deal damage to identified species
    if found_i != None:
        field[found_i][found_j].score = field[found_i][found_j].score - random.randint(10,60)
        
        
        
import time

record_field = np.zeros((100,100,51),dtype=float)
population_grid = mapfield(field)
record_field[:,:,0] = population_grid

for counter in range(1,51): # in this case I am conducting 50 iterations 
     
    # iterating through all cells on the battlefield
    for x in range(0,100):
        for y in range(0,100):
            # print("top tier iteration, i: "+str(i)+", j: "+str(j))
            # check if there is an agent within the respective cell
            if field[x][y] != None:
                # depending on the type: execute respective attack strategy
                if field[x][y].group == "A":
                    # one round of battle for this agent of type A
                    oneRoundAgentA(i = x, j = y,attackRange=ATTR)
                else: 
                    # one round of battle for this agent of type B
                    oneRoundAgentB(i = x, j = y,attackRange=ATTR)
    # identifying agents with life score of score or below - and removing them from the grid
    removeAgents(field)
    population_grid = mapfield(field)
    record_field[:,:,counter] = population_grid

    
fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(record_field[:,:,-1], cmap=colormap)
plt.title('Agent distribution: Group A green, B blue')
plt.show()
plt.savefig('/tmp/' + username + '/final.png')
plt.close()




A_list = np.zeros((51), dtype=int)
B_list = np.zeros((51), dtype=int)
time_array = np.linspace(0.0, 51.0, num=51)

for counter in range(0,51):
    A_list[counter]=np.count_nonzero(record_field[:,:,counter] == 1)
    B_list[counter]=np.count_nonzero(record_field[:,:,counter] == 2)
    
fig, ax = plt.subplots(figsize=(7,7))
plt.plot(time_array, A_list/NUMA, 'g', label='species A',linewidth=2)
plt.plot(time_array, B_list/NUMB, 'b', label='species B',linewidth=2)
plt.xlabel('time', fontsize=12)
plt.ylabel('species concentration', fontsize=16)
plt.legend()
plt.show()
plt.savefig('/tmp/' + username + '/time_series.png')
plt.close()




plt.rcParams["figure.figsize"] = (10, 10)
for counter in range(0,51):
    #display(plt.gcf())
    clear_output(wait=True)
    if counter == 0:
        myobj=plt.imshow(record_field[:,:,counter], cmap=colormap)
    else:
        myobj.set_data(record_field[:,:,counter])
    plt.show();

    plt.savefig('/tmp/' +username + '/frame%03d.png' % counter,format='png')    

plt.close()
os.system('convert -delay 20 /tmp/'  + username +'/frame*.png /tmp/' +username + '/animation.gif')
os.system('rm /tmp/' + username + '/frame*.png')
         
       
       
        
