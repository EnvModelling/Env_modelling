import numpy as np
from IPython.display import clear_output, display
import matplotlib.pyplot as plt #as per the above, much easier to write over and over again 
import time


def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4*M
    L += np.roll(M, (0,-1), (0,1)) # right neighbor
    L += np.roll(M, (0,+1), (0,1)) # left neighbor
    L += np.roll(M, (-1,0), (0,1)) # top neighbor
    L += np.roll(M, (+1,0), (0,1)) # bottom neighbor
    
    return L

def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """
    
    # Let's get the discrete Laplacians first
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)
    
    # Now apply the update formula
    diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
    diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t
    
    A += diff_A
    B += diff_B
    
    return A, B
    
def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """
    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))
    
    # Now let's add a disturbance in the center
    N2 = N//2
    radius = r = int(N/10.0)
    
    #A[N2-r:N2+r, N2-r:N2+r] = 0.50
    #B[N2-r:N2+r, N2-r:N2+r] = 0.25
    
    # PJC
    x=np.linspace(-100,100,N) 
    X,Y=np.meshgrid(x,x)    
    radius= np.sqrt(X*X+Y*Y) 
    ind=np.where(radius<10)
    A[ind]=1.
    B[ind]=0.0                                                                          
    
    return A, B
    
A, B = get_initial_configuration(200)

fig, ax = plt.subplots(1,2,figsize=(12,12))
first=ax[0].imshow(A, cmap='Greys',vmin=0, vmax=1)
second=ax[1].imshow(B, cmap='Greys',vmin=0, vmax=1)
ax[0].set_title('A')
ax[1].set_title('B')
ax[0].axis('off')
ax[1].axis('off')
fig.colorbar(first,ax=ax[0],fraction=0.046, pad=0.04)
fig.colorbar(first,ax=ax[1],fraction=0.046, pad=0.04)
plt.ion()
plt.show()
plt.close()


# update in time
delta_t = 1.0

# Diffusion coefficients
DA = 0.16
DB = 0.08

DA = 0.14
DB = 0.06


# define feed/kill rates
f = 0.060
k = 0.062

k = 0.065

# grid size
N = 200

# simulation steps
N_simulation_steps = 10000


A, B = get_initial_configuration(200)

for t in range(N_simulation_steps):
    A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
    
fig, ax = plt.subplots(1,2,figsize=(12,12))
ax[0].imshow(A, cmap='Greys',vmin=0, vmax=1)
ax[1].imshow(B, cmap='Greys',vmin=0, vmax=1)
ax[0].set_title('A')
ax[1].set_title('B')
ax[0].axis('off')
ax[1].axis('off')
fig.colorbar(first,ax=ax[0],fraction=0.046, pad=0.04)
fig.colorbar(first,ax=ax[1],fraction=0.046, pad=0.04)
plt.ion()
plt.show()
plt.close()

# We can check the values for both A and B 
print("Total amount of A = ", np.sum(A)) 
print("Total amount of B = ", np.sum(B))


# Experiment 2

# Create a 2D array of f values that increasesd as we move away from the centre.
x_values = np.linspace(-50.0, 50.0, num=200)
y_values = np.linspace(-50.0, 50.0, num=200)
z_values = np.zeros((200,200))
x_index=0
for x in x_values:
    y_index=0
    for y in y_values:
        z_values[x_index,y_index] = (np.exp(np.abs(x/50)*np.abs(y/50)))
        y_index+=1
    x_index+=1

# Diffusion coefficients
DA = 0.16
DB = 0.08

# define feed/kill rates
f = 0.030
k = 0.062
f_values = z_values*f
    
# Please insert the code required to run a simulation now using the variable 'f_values'

A, B = get_initial_configuration(200)

for t in range(N_simulation_steps):
    A, B = gray_scott_update(A, B, DA, DB, f_values, k, delta_t)
    
fig, ax = plt.subplots(1,2,figsize=(12,12))
ax[0].imshow(A, cmap='Greys',vmin=0, vmax=1)
ax[1].imshow(B, cmap='Greys',vmin=0, vmax=1)
ax[0].set_title('A')
ax[1].set_title('B')
ax[0].axis('off')
ax[1].axis('off')
fig.colorbar(first,ax=ax[0],fraction=0.046, pad=0.04)
fig.colorbar(first,ax=ax[1],fraction=0.046, pad=0.04)
plt.show()
plt.close()



# Experiment 3
# Diffusion coefficients
DA = 0.2
DB = 0.1

# define feed/kill rates
f = 0.035
k = 0.062

# mitosis
f=0.0367
k=0.0649



N_simulation_steps = 4000

delta_t = 1.0

# intialize the figures
A, B = get_initial_configuration(200)

fig, ax = plt.subplots(figsize=(12,12))
plt.imshow(A, cmap='Greys',vmin=0, vmax=1)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.close()

# Let us also store the total concentration of A and B as a function of time for an extra 
# plot after this simulation,

import os
import getpass

username=getpass.getuser()

A_conc_list=[]
B_conc_list=[]
time_list=[]
it=0
for i in range(N_simulation_steps):
    time_list.append(i)
    plt.rcParams["figure.figsize"] = (12, 12);
    #display(plt.gcf());
    clear_output(wait=True)
    A_new, B_new = gray_scott_update(A, B, DA, DB, f, k, delta_t)
    A_conc_list.append(np.sum(A_new))
    B_conc_list.append(np.sum(B_new))
    A, B = A_new, B_new
    
    if(np.mod(i,50)==0):
        it=it+1
        if(i==0):
            myobj=plt.imshow(A, cmap='Greys',vmin=0, vmax=1)
            #plt.colorbar(fraction=0.046, pad=0.04)
            #plt.show();
        else:
            myobj.set_data(A)
            
        myobj.set_clim((0.,1.))
        
        if not os.path.exists('/tmp/' + username):
            os.mkdir('/tmp/' + username)
        if it==0:
            os.system('rm /tmp/' + username + '/*')
        
        plt.savefig('/tmp/' +username + '/frame%03d.png' % it,format='png') 

os.system('convert -delay 20 /tmp/'  + username +'/frame*.png /tmp/' +username + '/animation.gif')
os.system('rm /tmp/' + username + '/frame*.png')

        
