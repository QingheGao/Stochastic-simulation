#!/usr/bin/env python
# coding: utf-8

# # Picture

# In[1504]:


@jit
def mandelbrot_set(re, im, max_iter):
    c = complex(re,im)
    z = 0.0j
    
    for i in range(max_iter):
        z=z*z+c
        if (np.abs(z)) > 2:
            return i
    return max_iter


# In[1505]:


columns = 1000
rows = 1000
result = np.zeros([rows,columns])
maxiter = 10000

for row_index , re in enumerate(np.linspace(-2.,1.,num = rows)):
    for columns_index , im in enumerate(np.linspace(-1.25,1.25,num = columns)):
        result[row_index,columns_index]=mandelbrot_set(re, im, maxiter)   


# In[1506]:


plt.figure(dpi=600)
plt.imshow(result.T, cmap='Blues', interpolation='bilinear', extent=[-2,1.,-1.25,1.25,])
plt.xlabel('Real')
plt.ylabel('Imaginary')
# plt.savefig('mandelbrot1.png',dpi=600)    
plt.show()


# In[1507]:


def histogram_colouring():
    columns = 1000
    rows = 1000
    IterationCounts = np.zeros((rows,columns))
    NumIterationsPerPixel = np.zeros(maxiter+1)
    

    for x in range(rows):
        for y  in range(columns):
            i = result[x][y]
            NumIterationsPerPixel[int(i)] += 1

    total = 0;
    for i in range(maxiter):
        total += NumIterationsPerPixel[i]

    hue = np.zeros((rows,columns))
    for x in range(rows):
        for y in range(columns):
            iteration = result[x][y]

            for i in range(int(iteration)):
                hue[x][y] += NumIterationsPerPixel[i] / total
            
    return(hue)


# In[1508]:


# Picture of Mandelbrot set for different iteration number
hue = histogram_colouring()


# In[1509]:


plt.figure(dpi=600)
plt.imshow(hue.T, cmap='Blues', interpolation='bilinear', extent=[-2,1.,-1.25,1.25,])
plt.xlabel('Real')
plt.ylabel('Imaginary')
# plt.savefig('mandelbrot1.png',dpi=600)    
plt.show()


# ## True vaule

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from numba import jit


# In[1174]:


@jit
def mandelbrot_set(re, im, max_iter):
    c = complex(re,im)
    z = 0.0j
    
    for i in range(max_iter):
        z=z*z+c
        if (np.abs(z)) > 2:
            return 0.
    return 1.


# In[1175]:


def f_random():
    
    f_arealist = []
    maxiter = 5000
    s_num = 20000
    f_simulationlist=np.arange(1,5000,1)

    for i in f_simulationlist:
        area = 0
        for j in range(s_num):
            u1 = np.random.uniform(-2,2)
            u2 = np.random.uniform(-2,2)
            area += mandelbrot_set(u1,u2,maxiter)#evaluation if it is 1
        norm = (2 + 2) * (2 * 2) / (s_num)
        f_arealist.append(area * norm)
        
    return f_simulationlist,f_arealist


# In[1176]:


f_simulationlist, f_arealist= f_random()
True_Ais=np.mean(f_arealist)


# In[1177]:


plt.plot(f_simulationlist,f_arealist)
print('True mean area: ', True_Ais)
print('Variance: ', np.var(f_arealist))
# print('Std: ', np.std(f_arealist))
plt.xlabel('Simulation times')
plt.ylabel('True Area: F_Ais')
plt.show()


# ## Generate sample pic

# In[551]:


samples = 4
samples_x = 4*np.random.rand(samples)-2
samples_y = 4*np.random.rand(samples)-2

min_limit = -2
max_limit = 2
fig = plt.figure(figsize=(6,6))     
plt.scatter(samples_x,samples_y,c='r')
plt.xlim(min_limit,max_limit)
plt.ylim(min_limit,max_limit)
plt.xlabel("Pure Random Sampling")
plt.savefig('randomsampling.jpg',dpi=600)

plt.show()


# ## Different sample number of full random

# In[432]:


#change sample
def s_random():

    s_randomlist = np.arange(1000,10000,100)
    maxiter = 1000
    simulation_list= 100
    total_dis=[]

        
    for num_iterations in s_randomlist:
        s_arealist = []
        for i in range(simulation_list) :
            area = 0
            for j in range(num_iterations):
                u1 = np.random.uniform(-2,2)
                u2 = np.random.uniform(-2,2)
                area += mandelbrot_set(u1,u2,maxiter)#evaluation if it is 1
            norm = (2 + 2) * (2 * 2) / (num_iterations)
            s_arealist.append(area * norm)
        total_dis.append(s_arealist)
        
    return s_randomlist,s_arealist,total_dis


# In[433]:


s_randomlist,s_arealist,total_dis = s_random()


# In[434]:


meanlist=[]
varlist=[]
stdlist=[]
fill1list=[]
fill2list=[]

for i in range(len(total_dis)):
    meanlist.append(np.mean(total_dis[i]))
    varlist.append(np.var(total_dis[i]))
    stdlist.append(np.std(total_dis[i]))
    fill1list.append(np.mean(total_dis[i])+1*np.std(total_dis[i]))
    fill2list.append(np.mean(total_dis[i])-1*np.std(total_dis[i]))
# plt.subplot(3,3,1)
# plt.hist(total_dis[0], bins=20,color="#13eac9")
# plt.xlabel('area')
# plt.ylabel('number')


# In[576]:


plt.plot(s_randomlist,meanlist,label='Mean Area')
plt.fill_between(s_randomlist,fill1list,fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('PR_Sample Number,s')
plt.ylim(1.3,1.7)
plt.ylabel('Area')
plt.legend()
plt.rcParams['figure.figsize'] = (7.0, 5.0) 
plt.savefig('pure_random_change_S_conver.jpg',dpi=600)


# # Varying iteration number

# ## Full random Ais

# In[923]:


def i_random():
    itera_list = np.arange(5,5000,10)
    maxrandom = 10000
    i_arealist = []
#     F_i_simulationlist=10
#     total_F_i_area=[]
    F_i_arealist = []
    for maxiter in itera_list:
#         F_i_arealist = []
#         for i in range(F_i_simulationlist):
        area=0
        for j in range(maxrandom):
            u1 = np.random.uniform(-2,2)
            u2 = np.random.uniform(-2,2)
            area += mandelbrot_set(u1,u2,maxiter)#evaluation if it is 1
        norm = (2 + 2) * (2 * 2) / (maxrandom)
        F_i_arealist.append(area * norm)
#     total_F_i_area.append(F_i_arealist)
        
        
    return itera_list,F_i_arealist


# In[924]:


itera_list,F_i_arealist = i_random()


# In[926]:


# plt.plot(randomlist,[np.mean(arealist),np.mean(arealist)],'k--')
plt.figure(figsize=(10, 5))
plt.plot(itera_list,F_i_arealist)
# plt.fill_between(F_i_list,F_i_fill1list,F_i_fill2list,color='black',alpha=0.1,label='Std')
# plt.errorbar(F_i_list, F_i_meanlist, yerr=F_i_stdlist,fmt='-o')
plt.xlabel('Iteration Numbers, i')
plt.ylabel('Area')
# plt.savefig('pure_random_change_i_area.jpg',dpi=600)
plt.show()


# In[1178]:


Coverageilist=[]
for i in F_i_arealist:
    h=np.abs(i-True_Ais)
    Coverageilist.append(h)
plt.figure(figsize=(10, 5))
plt.plot(itera_list,Coverageilist)
plt.xlabel('Iteration Numbers, i')
plt.ylabel('Convergence rate')
# plt.savefig('pure_random_change_i_conver.jpg',dpi=600)
plt.show()


# ## Accuracy 
# 

# In[1138]:


def F_r_random():
    
    F_r_arealist = []
    maxiter = 1000
    s_num = 10000
    F_r_simulationlist=np.arange(0,1000,1)

    for i in F_r_simulationlist:
        area = 0
        for j in range(s_num):
            u1 = np.random.uniform(-2,2)
            u2 = np.random.uniform(-2,2)
            area += mandelbrot_set(u1,u2,maxiter)#evaluation if it is 1
        norm = (2 + 2) * (2 * 2) / (s_num)
        F_r_arealist.append(area * norm)
        
    return F_r_simulationlist,F_r_arealist


# In[1139]:


F_r_simulationlist, F_r_arealist= F_r_random()


# In[1199]:


plt.plot(F_r_simulationlist,F_r_arealist)
print('Mean area of fully random: ', np.mean(F_r_arealist))
print('Variance area of fully random: ', np.var(F_r_arealist))
print('STD of fully random: ', np.std(F_r_arealist))
plt.xlabel('Simulation times')
plt.ylabel('Area')
plt.show()


# # Latin Hypercube sampling

# ### area True value of LHS

# In[224]:


import numpy as np
import matplotlib.pyplot as plt
from numba import jit


# ## Generate sample pic

# In[555]:


from smt.sampling_methods import LHS
n=4
min_limit = -2
max_limit = 2
    # Create LHS sampling
xlimits = np.array([[-2,2], [-2,2]])
sampling = LHS(xlimits=xlimits)
sample=sampling(n)
fig = plt.figure(figsize=(6,6))         
plt.xlim(min_limit,max_limit)
plt.ylim(min_limit,max_limit)
plt.scatter(sample[:,0],sample[:,1],c='r')

for i in np.linspace(min_limit,max_limit,num=n+1):
    plt.axvline(i,linewidth=0.8)
    plt.axhline(i,linewidth=0.8)
# for i in np.linspace(min_limit,max_limit,3):
#     plt.axvline(i,color='green')
#     plt.axhline(i,color='green')
plt.xlabel("Latin Hypercube Sampling")
plt.savefig('LHSampling.jpg',dpi=600)
plt.show()


# In[225]:


@jit
def mandelbrot_set(re, im, max_iter):
    c = complex(re,im)
    z = 0.0j
    
    for i in range(max_iter):
        z=z*z+c
        if (np.abs(z)) > 2:
            return 0.
    return 1.


# ### Change number of samples

# In[428]:


##change s

def s_LHS():
    from smt.sampling_methods import LHS
    
    lhs_simulationlist=100

    # Generate sample numbers using LHSL sampling
    xlimits = np.array([[-2,2], [-2,2]])
    sampling = LHS(xlimits=xlimits)
    total_LHS_arealist=[]
    maxiter = 1000
    max_s = 10000
    min_s = 1000
    ts_s = 100

    # Generate a different amount of sample numbers
    L_srandomlist = np.arange(min_s,max_s,ts_s)

    # Calculate Mandelbrot for different amount of sample numbers
    for num_iterations in L_srandomlist:
        LHS_s_arealist = []
        for h in range(lhs_simulationlist):
            area = 0
            LHS = sampling(num_iterations)
            for i in range(num_iterations): 
                u1 = LHS[i][0]
                u2 = LHS[i][1]

                # Evaluation if it is 1
                area += mandelbrot_set(u1,u2,maxiter) 
            norm = (2 + 2) * (2 * 2) / (num_iterations)
            LHS_s_arealist.append(area * norm)
        total_LHS_arealist.append(LHS_s_arealist)
    
    return(L_srandomlist,LHS_s_arealist,total_LHS_arealist)


# In[429]:


L_srandomlist,LHS_s_arealist,total_LHS_arealist = s_LHS()


# In[430]:


LHS_meanlist=[]
LHS_varlist=[]
LHS_stdlist=[]
LHS_fill1list=[]
LHS_fill2list=[]

for i in range(len(total_LHS_arealist)):
    LHS_meanlist.append(np.mean(total_LHS_arealist[i]))
    LHS_varlist.append(np.var(total_LHS_arealist[i]))
    LHS_stdlist.append(np.std(total_LHS_arealist[i]))
    LHS_fill1list.append(np.mean(total_LHS_arealist[i])+1*np.std(total_LHS_arealist[i]))
    LHS_fill2list.append(np.mean(total_LHS_arealist[i])-1*np.std(total_LHS_arealist[i]))


# In[578]:


plt.plot(L_srandomlist,LHS_meanlist,label='Mean Area')
plt.fill_between(L_srandomlist,LHS_fill1list,LHS_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('LHS_Sample number')
plt.ylabel('Area')
plt.ylim(1.3,1.7)
plt.legend()
plt.rcParams['figure.figsize'] = (7.0, 5.0) 
plt.savefig('LHS_change_S_conver.jpg',dpi=600)


# ### Changing iteration 

# In[767]:


def i_LHS():
    from smt.sampling_methods import LHS

    # Create LHS sampling
    xlimits = np.array([[-2,2], [-2,2]])
    sampling = LHS(xlimits=xlimits)

    L_itera_list = np.arange(5,5000,10)
    s_num = 10000
    
    L_i_arealist = []

    # Calculate the mandelbrot set for different iteration number
    for maxiter in L_itera_list:
        area = 0
        LHS = sampling(s_num)
        for i in range(s_num):  
            u1 = LHS[i][0]
            u2 = LHS[i][1]

            # Evaluation if it is 1
            area += mandelbrot_set(u1,u2,maxiter)
        norm = (2 + 2) * (2 * 2) / (s_num)
        L_i_arealist.append(area * norm)

    return L_itera_list, L_i_arealist


# In[768]:


L_itera_list, L_i_arealist = i_LHS()


# In[769]:


plt.plot(L_itera_list,L_i_arealist)
print('Mean area: ', np.mean(L_i_arealist))
print('Variance area: ', np.var(L_i_arealist))
plt.xlabel('LHS_Iteration number, i')
plt.ylabel('Area')
# plt.savefig('LHS_change_i_area.jpg',dpi=600)
plt.show()


# In[1179]:


L_i_Coverageilist=[]
for i in L_i_arealist:
    h=(i-True_Ais)
    L_i_Coverageilist.append(h)
plt.figure(figsize=(10, 5))
plt.plot(L_itera_list,L_i_Coverageilist)
plt.xlabel('Iteration numbers, i')
plt.ylabel('Convergence rate')
# plt.savefig('LHS_change_i_conver.jpg',dpi=600)
plt.show()


# ### LHS_accuracy

# In[1144]:


### For more simulation
def v_LHS():
    from smt.sampling_methods import LHS

    # Create LHS sampling
    xlimits = np.array([[-2,2], [-2,2]])
    sampling = LHS(xlimits=xlimits)
    maxiter = 1000
    s_num = 10000
    
    v_L_arealist = []
    v_L_simulationlist=np.arange(0,1000,1)

    # Calculate the mandelbrot set for different iteration number
    for z in v_L_simulationlist:
        area = 0
        LHS = sampling(s_num)
        for i in range(s_num):  
            u1 = LHS[i][0]
            u2 = LHS[i][1]
            # Evaluation if it is 1
            area += mandelbrot_set(u1,u2,maxiter)
        norm = (2 + 2) * (2 * 2) / (s_num)
        v_L_arealist.append(area * norm)

    return v_L_simulationlist, v_L_arealist


# In[1145]:


v_L_simulationlist, v_L_arealist= v_LHS()


# In[1200]:


plt.plot(v_L_simulationlist,v_L_arealist)
print('Mean area of LHS: ', np.mean(v_L_arealist))
print('Variance area of LHS: ', np.var(v_L_arealist))
print('STD area of LHS: ', np.std(v_L_arealist))
plt.xlabel('Simulation times')
plt.ylabel('Area')
plt.show()


# # Orthogonal sampling

# ### True value

# In[161]:


import numpy as np
import random
import matplotlib.pyplot as plt
from numba import jit


# In[565]:


n_per_dim = 2
X,Y,n_range,n_subset_range = orthogonal_sampling(n_per_dim)
fig = plt.figure(figsize=(6,6))         
# plt.xlim(min_limit,max_limit)
# plt.ylim(min_limit,max_limit)
plt.scatter(X,Y,c='r')

min_fig = -2
max_fig=2
for i in n_range:
    plt.axhline(i,linewidth=0.8)
    plt.axvline(i,linewidth=0.8)
# for i in n_subset_range:
#     plt.axhline(i,color='black')
#     plt.axvline(i,color='black')
# plt.axhline(min_fig,color='black')
# plt.axhline(max_fig,color='black')
# plt.axvline(0)
# plt.axvline(min_fig,color='black')
# plt.axvline(max_fig,color='black')
plt.axhline()
plt.xlabel("Orthogonal Sampling")
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.savefig('Orthogonalsampling.jpg',dpi=600)
plt.show()


# In[562]:


@jit
def mandelbrot_set(re, im, max_iter):
    c = complex(re,im)
    z = 0.0j
    
    for i in range(max_iter):
        z=z*z+c
        if (np.abs(z)) > 2:
            return 0.
    return 1.


# In[416]:


#orthogonal sampling

def orthogonal_sampling(n_per_dim):
    min_fig=-2.0
    max_fig=2.0
    n_subsets = int(n_per_dim)
    n = int(n_per_dim**2)
    n_range = np.linspace(min_fig,max_fig,n+1)
    subsetrange = int(n/n_subsets)
    n_subset_range = np.linspace(min_fig,max_fig,n_subsets+1)
    in_use_x = np.zeros(n)
    in_use_y = np.zeros(n)
    X = []
    Y = []
    for subset_x in range(n_subsets):
        for subset_y in range(n_subsets):
            x_options = in_use_x[subset_x*subsetrange:(subset_x*subsetrange)+subsetrange]==0
            free_options_x = [i+subset_x*subsetrange for i, x in enumerate(x_options) if x]
            row_choice = np.random.choice(free_options_x)
            x_choice = np.random.uniform(high=n_range[row_choice+1],low=n_range[row_choice])
            in_use_x[row_choice]=1
            X.append(x_choice)

            y_options = in_use_y[subset_y*subsetrange:(subset_y*subsetrange)+subsetrange]==0
            free_options_y = [i+subset_y*subsetrange for i, y in enumerate(y_options) if y]
            col_choice = np.random.choice(free_options_y)
            y_choice = np.random.uniform(high=n_range[col_choice+1],low=n_range[col_choice])
            in_use_y[col_choice]=1
            Y.append(y_choice)

    return [X,Y,n_range,n_subset_range]


# ### Change sample number

# In[425]:


#change sample number 
o_s_simulationlist=100
maxiter=1000
o_s_samplelist= np.arange(1000,10000,100)
o_s_arealist = []
total_o_s_arealist=[]

for sample in o_s_samplelist:
    o_s_arealist = []
    for h in range(o_s_simulationlist):
        n_per_dim=int(np.sqrt(sample))
        area = 0
        X,Y,n_range,n_subset_range = orthogonal_sampling(n_per_dim)
        for i in range(len(X)):
            area += mandelbrot_set(X[i],Y[i],maxiter) 
        norm = (2*2) * (2 * 2) / (sample)
        o_s_arealist.append(area * norm)
    total_o_s_arealist.append(o_s_arealist)


# In[426]:


OS_meanlist=[]
OS_varlist=[]
OS_stdlist=[]
OS_fill1list=[]
OS_fill2list=[]

for i in range(len(total_o_s_arealist)):
    OS_meanlist.append(np.mean(total_o_s_arealist[i]))
    OS_varlist.append(np.var(total_o_s_arealist[i]))
    OS_stdlist.append(np.std(total_o_s_arealist[i]))
    OS_fill1list.append(np.mean(total_o_s_arealist[i])+1*np.std(total_o_s_arealist[i]))
    OS_fill2list.append(np.mean(total_o_s_arealist[i])-1*np.std(total_o_s_arealist[i]))


# In[583]:


plt.plot(o_s_samplelist,OS_meanlist,label='Mean Area')
plt.fill_between(o_s_samplelist,OS_fill1list,OS_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('OS_Sample Number,s')
plt.ylabel('Area')
plt.ylim(1.3,1.7)
plt.legend()
plt.rcParams['figure.figsize'] = (7.0, 5.0) 
plt.savefig('OS_change_S_conver.jpg',dpi=600)


# ### Change iteration times

# In[774]:


#change iteration time
o_i_maxiterlist=np.arange(5,5000,10)

s_samplelist= 10000

o_i_arealist = []

for maxiter in o_i_maxiterlist:
    n_per_dim=int(np.sqrt(s_samplelist))
    area = 0
    X,Y,n_range,n_subset_range = orthogonal_sampling(n_per_dim)
    for i in range(len(X)):
        area += mandelbrot_set(X[i],Y[i],maxiter) 
    norm = (2*2) * (2 * 2) / (s_samplelist)
    o_i_arealist.append(area * norm)
    


# In[775]:


O_i_Ajs=np.mean(o_i_arealist)
plt.plot(o_i_maxiterlist,o_i_arealist)
plt.xlabel('Iteration times,i')
plt.ylabel('Area')
print('mean',np.mean(O_i_Ajs))
print('variance',np.var(o_i_arealist))
# plt.savefig('OS_change_i_area.jpg',dpi=600)


# In[1180]:


#Ais
o_i_Coverageilist=[]
for i in o_i_arealist:
    h=np.abs(i-True_Ais)
    o_i_Coverageilist.append(h)
plt.figure(figsize=(10, 5))
plt.plot(o_i_maxiterlist,o_i_Coverageilist)
plt.xlabel('Number of iteration numbers, i')
plt.ylabel('Converage rate')
# plt.savefig('OS_change_i_conver.jpg',dpi=600)
plt.show()


# In[777]:


#Fixed s and i 
#simulation many time:
sample= 10000
maxiter=1000
n_per_dim=int(np.sqrt(sample))
simulationlist=np.arange(0,1000,1)
arealist=[]

for i in simulationlist:
    area = 0
    X,Y,n_range,n_subset_range = orthogonal_sampling(n_per_dim)
    for i in range(len(X)):
        area += mandelbrot_set(X[i],Y[i],maxiter) 
    norm = (2*2) * (2 * 2) / (n_per_dim*n_per_dim)
    arealist.append(area * norm)


# In[1201]:


plt.plot(simulationlist,arealist)
plt.xlabel('Simulation times')
plt.ylabel('Area')
print('mean',np.mean(arealist))
print('variance',np.var(arealist))
print('STD',np.std(arealist))


# # Control Variables

# In[451]:


import random


def generate_control(n1,n2,num):
    x=np.random.uniform(n1,n2,num)
    y=np.random.uniform(n1,n2,num)
    cov=np.cov(x,y)[0][1]
    var=np.var(y)
    c=-cov/var
    newvar=x+c*(y-np.mean(y))
    sample_list=[]
    for x1 in newvar:
        if -2<= x1 <=2:
            sample_list.append(x1)
    return sample_list



# ### Change sample numbers

# In[456]:


def s_control():
    co_simulationlist=100
    maxiter = 1000
    n1=-2
    n2=2
    co_num=np.arange(1000,10000,100)
    total_co_arealist=[]
    for h in co_num:
        co_arealist=[]
        for i in range(co_simulationlist):
            x=generate_control(n1,n2,h)
            y=generate_control(n1,n2,h)
            min_num=np.minimum(len(x),len(y))
            x_random=random.sample(x,min_num)
            y_random=random.sample(y,min_num)
            area = 0
            for j in range(len(x_random)):
                u1 = x_random[j]
                u2 = y_random[j]
                area += mandelbrot_set(u1 ,u2,maxiter) 
            norm = (4) * (2 * 2) / (len(x_random))
            co_arealist.append(area * norm)
        total_co_arealist.append(co_arealist)
        
    return co_num,co_arealist,total_co_arealist


# In[457]:


co_num,co_arealist,total_co_arealist=s_control()


# In[458]:


co_meanlist=[]
co_varlist=[]
co_stdlist=[]
co_fill1list=[]
co_fill2list=[]

for i in range(len(total_co_arealist)):
    co_meanlist.append(np.mean(total_co_arealist[i]))
    co_varlist.append(np.var(total_co_arealist[i]))
    co_stdlist.append(np.std(total_co_arealist[i]))
    co_fill1list.append(np.mean(total_co_arealist[i])+1*np.std(total_co_arealist[i]))
    co_fill2list.append(np.mean(total_co_arealist[i])-1*np.std(total_co_arealist[i]))


# In[587]:


plt.plot(co_num,co_meanlist,label='Mean area')
plt.fill_between(co_num,co_fill1list,co_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('Co_sample number,s')
plt.ylim(1.3,1.7)
plt.ylabel('Area')
plt.legend()
plt.savefig('CO_change_S_conver.jpg',dpi=600)
plt.rcParams['figure.figsize'] = (7.0, 5.0) 


# ### Change iteration times

# In[476]:


def i_co_control():
    i_co_maxiterlist = np.arange(5,5000,10)
    n1=-2
    n2=2
    co_num=10000
    i_co_arealist=[]
    for maxiter in i_co_maxiterlist:
        x=generate_control(n1,n2,co_num)
        y=generate_control(n1,n2,co_num)
        min_num=np.minimum(len(x),len(y))
        x_random=random.sample(x,min_num)
        y_random=random.sample(y,min_num)
        area = 0
        for j in range(len(x_random)):
            u1 = x_random[j]
            u2 = y_random[j]
            area += mandelbrot_set(u1 ,u2,maxiter) 
        norm = (4) * (2 * 2) / (len(x_random))
        i_co_arealist.append(area * norm)
        
    return i_co_maxiterlist,i_co_arealist


# In[477]:


i_co_maxiterlist,i_co_arealist=i_co_control()


# In[593]:


co_i_Ajs=np.mean(i_co_arealist)
plt.plot(i_co_maxiterlist,i_co_arealist)
plt.xlabel('Co_iteration number,i')
plt.ylabel('Area')
print('mean',np.mean(co_i_Ajs))
print('variance',np.var(i_co_arealist))
plt.savefig('CO_change_i_area.jpg',dpi=600)


# In[1181]:


co_i_Coverageilist=[]

for i in i_co_arealist:
    h=np.abs(i-True_Ais)
    co_i_Coverageilist.append(h)
plt.figure(figsize=(10, 5))
plt.plot(i_co_maxiterlist,co_i_Coverageilist)
plt.xlabel('Iteration numbers, i')
plt.ylabel('Converage rate')
plt.savefig('CO_change_i_conver.jpg',dpi=600)
plt.show()


# ### Accuracy

# In[782]:


def A_co_control():
    maxiter = 1000
    A_co_simulationlist=1000
    n1=-2
    n2=2
    co_num=10000
    A_co_arealist=[]
    for i in range(A_co_simulationlist):
        x=generate_control(n1,n2,co_num)
        y=generate_control(n1,n2,co_num)
        min_num=np.minimum(len(x),len(y))
        x_random=random.sample(x,min_num)
        y_random=random.sample(y,min_num)
        area = 0
        for j in range(len(x_random)):
            u1 = x_random[j]
            u2 = y_random[j]
            area += mandelbrot_set(u1 ,u2,maxiter) 
        norm = (4) * (2 * 2) / (len(x_random))
        A_co_arealist.append(area * norm)
        
    return A_co_simulationlist,A_co_arealist


# In[783]:


A_co_simulationlist,A_co_arealist= A_co_control()


# In[1202]:


plt.plot(np.arange(0,A_co_simulationlist,1),A_co_arealist)
print(np.mean(A_co_arealist))
print(np.var(A_co_arealist))
print(np.std(A_co_arealist))


# # Sobol 

# In[ ]:


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


# In[931]:


def sabol(num):
    problem = {
        'num_vars': 2,
        'names': ['x1','x2'],
        'bounds': [[-2, 2],
                  [-2, 2]]
    }
    param_values = saltelli.sample(problem, num)
    return param_values


# ### Change sample numbers

# In[1206]:


num=np.arange(100,5000,5)
maxiter=1000

sa_arealist=[]
total_sa_arealist=[]

for z in num:
    area = 0
    sabl=sabol(z)
    for i in range(len(sabl)):
        u1=sabl[i][0]
        u2=sabl[i][1]
        area += mandelbrot_set(u1,u2,maxiter) 
    norm = (2*2) * (2 * 2) / (len(sabl))
    sa_arealist.append(area * norm)


# In[1208]:


plt.rcParams['figure.figsize'] = (7, 5.0) 
plt.plot(num*6,sa_arealist)
plt.xlabel('Sample number,s')
plt.ylabel('Area')
plt.savefig('Sobol_sequence_sample.jpg',dpi=600)


# ### Change iteration numbers

# In[1083]:


num=5000
maxiter=1000
so_simulationlis=1000
true_so_arealist=[]

for i in range(so_simulationlis):
    area = 0
    sabl=sabol(num)
    for i in range(len(sabl)):
        u1=sabl[i][0]
        u2=sabl[i][1]
        area += mandelbrot_set(u1,u2,maxiter) 
    norm = (2*2) * (2 * 2) / (len(sabl))
    true_so_arealist.append(area * norm)


# In[1203]:


plt.plot(np.arange(0,so_simulationlis,1),true_so_arealist)
print(np.mean(true_so_arealist))
print(np.var(true_so_arealist))
print(np.std(true_so_arealist))


# In[971]:


num=1666
maxiterlist=np.arange(5,5000,10)

ia_arealist=[]

for maxiter in maxiterlist:
    area = 0
    sabl=sabol(num)
    for i in range(len(sabl)):
        u1=sabl[i][0]
        u2=sabl[i][1]
        area += mandelbrot_set(u1,u2,maxiter) 
    norm = (2*2) * (2 * 2) / (len(sabl))
    ia_arealist.append(area * norm)


# In[1074]:


plt.plot(maxiterlist,ia_arealist)
plt.xlabel('So_iteration number,i')
plt.ylabel('Area')


# In[1182]:


plt.plot(maxiterlist,np.abs(ia_arealist-True_Ais))
plt.xlabel('So_iteration number,i')
plt.ylabel('Absolute error')
plt.savefig('Sobol_i_cov.jpg',dpi=600)


# In[1130]:


plt.rcParams['figure.figsize'] = (10.0, 5.0) 
n=4
h=sabol(n)
plt.subplot(1,2,1)
plt.scatter(h[:,0],h[:,1])
plt.xlabel('Sobol sequence')
plt.subplot(1,2,2)
plt.scatter(np.random.uniform(-2,2,n*6),np.random.uniform(-2,2,n*6))
plt.xlabel('Fully random')
plt.legend()
plt.savefig('Sobol sequence.jpg',dpi=600)


# # Poisson disk sampling

# In[1592]:


from bridson import poisson_disc_samples


# ### Poisson sampling

# In[1593]:


num=200

r = np.sqrt(2)*np.sqrt((16/np.pi)/num)
poss=poisson_disc_samples(width=4, height=4, r=r)
x=[x-2 for x,y in poss]
y=[y-2 for x,y in poss]

plt.scatter(x,y,s=10)
plt.scatter(x, y, color='', marker='o', edgecolors='g', s=100)
plt.xlabel('Possion disk sampling')
plt.savefig('Pos_sampling.jpg',dpi=600)


# ### Change sample numbers

# In[1468]:


ponumlist=np.arange(1260,12600,100)
maxiter=1000
pd_simulationlis=100
s_pd_arealist=[]
total_pd_arealist=[]
S_length=[]
for h in ponumlist:
    s_pd_arealist=[]
    for i in range(pd_simulationlis):
        area=0
        r = np.sqrt(2)*np.sqrt((16/np.pi)/num)
        poss=poisson_disc_samples(width=4, height=4, r=r)
        x=[x-2 for x,y in poss]
        y=[y-2 for x,y in poss]
        for h in range(len(x)):
            u1=x[h]
            u2=y[h]
            area += mandelbrot_set(u1,u2,maxiter) 
        norm = (2*2) * (2 * 2) / (len(x))
        s_pd_arealist.append(area * norm)
    S_length.append(len(x))
    total_pd_arealist.append(s_pd_arealist)


# In[1471]:


PO_meanlist=[]
PO_varlist=[]
PO_stdlist=[]
PO_fill1list=[]
PO_fill2list=[]

for i in range(len(total_pd_arealist)):
    PO_meanlist.append(np.mean(total_pd_arealist[i]))
    PO_varlist.append(np.var(total_pd_arealist[i]))
    PO_stdlist.append(np.std(total_pd_arealist[i]))
    PO_fill1list.append(np.mean(total_pd_arealist[i])+1*np.std(total_pd_arealist))
    PO_fill2list.append(np.mean(total_pd_arealist[i])-1*np.std(total_pd_arealist))


# In[1499]:


plt.plot(ponumlist,PO_meanlist,label='Mean area')
plt.fill_between(ponumlist,PO_fill1list,PO_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('Sample number,s')
plt.ylim(1.3,1.7)
plt.xlim(1000,10000)
plt.ylabel('Area')
plt.legend()
plt.savefig('PO_change_S_conver.jpg',dpi=600)
plt.rcParams['figure.figsize'] = (7.0, 5.0) 


# ### Change iteration numbers

# In[1494]:


ponumm=12600
maxiterlist=np.arange(5,5000,10)

i_pd_arealist=[]

for maxiter in maxiterlist:
    area=0
    r = np.sqrt(2)*np.sqrt((16/np.pi)/num)
    poss=poisson_disc_samples(width=4, height=4, r=r)
    x=[x-2 for x,y in poss]
    y=[y-2 for x,y in poss]
    for h in range(len(x)):
        u1=x[h]
        u2=y[h]
        area += mandelbrot_set(u1,u2,maxiter) 
    norm = (2*2) * (2 * 2) / (len(x))
    i_pd_arealist.append(area * norm)


# In[1495]:


plt.plot(maxiterlist,i_pd_arealist)
plt.xlabel('Iteration number,i')
plt.ylabel('Area')


# In[1497]:


plt.plot(maxiterlist,np.abs(i_pd_arealist-True_Ais))
plt.xlabel('Iteration number,i')
plt.ylabel('Covergence rate')
plt.savefig('Pos_i_cov.jpg',dpi=600)


# In[1492]:


num=12600
maxiter=1000
pd_simulationlis=1000
true_pd_arealist=[]
length=[]
for i in range(pd_simulationlis):
    area=0
    r = np.sqrt(2)*np.sqrt((16/np.pi)/num)
    poss=poisson_disc_samples(width=4, height=4, r=r)
    x=[x-2 for x,y in poss]
    y=[y-2 for x,y in poss]
    length.append(len(x))
    for h in range(len(x)):
        u1=x[h]
        u2=y[h]
        area += mandelbrot_set(u1,u2,maxiter) 
    norm = (2*2) * (2 * 2) / (len(x))
    true_pd_arealist.append(area * norm)


# In[1493]:


print(np.mean(true_pd_arealist))
print(np.var(true_pd_arealist))


# ### Changing s 

# In[1197]:


plt.rcParams['figure.figsize'] = (10.0, 7.0) 
plt.subplot(2,2,1)
plt.plot(s_randomlist,meanlist,label='PR_Mean Area')
plt.fill_between(s_randomlist,fill1list,fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
# plt.xlabel('Sample Number,s')
plt.ylim(1.3,1.7)
plt.ylabel('Area')
plt.legend()
plt.subplot(2,2,2)
plt.plot(L_srandomlist,LHS_meanlist,label='LHS_Mean Area')
plt.fill_between(L_srandomlist,LHS_fill1list,LHS_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
# plt.xlabel('Sample number')
# plt.ylabel('Area')
plt.ylim(1.3,1.7)
plt.legend()
plt.subplot(2,2,3)
plt.plot(o_s_samplelist,OS_meanlist,label='OS_Mean Area')
plt.fill_between(o_s_samplelist,OS_fill1list,OS_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('Sample Number,s')
plt.ylabel('Area')
plt.ylim(1.3,1.7)
plt.legend()
plt.subplot(2,2,4)
plt.plot(co_num,co_meanlist,label='CO_Mean area')
plt.fill_between(co_num,co_fill1list,co_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.xlabel('sample number,s')
plt.ylim(1.3,1.7)
# plt.ylabel('Area')
plt.legend()
plt.savefig('Changing_s.jpg',dpi=600)
plt.show()


# In[1198]:


plt.rcParams['figure.figsize'] = (7, 5.0) 
plt.plot(num*6,sa_arealist)
plt.xlabel('So_sample number,s')
plt.ylabel('Area')
# plt.savefig('Sobol sequence sample.jpg',dpi=600)


# ### Accuracy analysis (variance)

# In[1186]:


plt.rcParams['figure.figsize'] = (10.0, 7.0) 
plt.subplot(2,2,1)
plt.plot(s_randomlist,varlist,label='FRS')
plt.xlabel('Sample number, s')
plt.ylim(0,0.02)
plt.ylabel('Variance')
plt.legend()
plt.subplot(2,2,2)
plt.plot(L_srandomlist,LHS_varlist,label='LHS')
plt.xlabel('Sample number, s')
# plt.ylabel('Converage rate')
plt.ylim(0,0.02)
plt.legend()
plt.subplot(2,2,3)
plt.plot(o_s_samplelist,OS_varlist,label='OS')
plt.xlabel('Sample number, s')
plt.ylabel('Variance')
plt.ylim(0,0.02)
plt.legend()
plt.subplot(2,2,4)
plt.plot(co_num,co_varlist,label='CV')
plt.xlabel('Sample number, s')
# plt.ylabel('Converage rate')
plt.ylim(0,0.02)
plt.legend()
plt.savefig('Fourmethod_variance.jpg',dpi=600)
plt.show()


# In[1502]:


plt.plot(ponumlist,PO_varlist,label='Mean area')
# plt.fill_between(ponumlist,PO_fill1list,PO_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.ylabel('Variance')
plt.xlim(1000,10000)
plt.xlabel('Sample number,s')
plt.legend()
plt.savefig('PO_change_var.jpg',dpi=600)
plt.rcParams['figure.figsize'] = (7.0, 5.0) 


# ### Accuracy analysis (STD)

# In[1190]:


plt.rcParams['figure.figsize'] = (10.0, 7.0) 
plt.subplot(2,2,1)
plt.plot(s_randomlist,stdlist,label='FRS')
plt.xlabel('Sample number, s')
plt.ylim(0,0.2)
plt.ylabel('Stander Deviation')
plt.legend()
plt.subplot(2,2,2)
plt.plot(L_srandomlist,LHS_stdlist,label='LHS')
plt.xlabel('Sample number, s')
# plt.ylabel('Converage rate')
plt.ylim(0,0.2)
plt.legend()
plt.subplot(2,2,3)
plt.plot(o_s_samplelist,OS_stdlist,label='OS')
plt.xlabel('Sample number, s')
plt.ylabel('Stander Deviation')
plt.ylim(0,0.2)
plt.legend()
plt.subplot(2,2,4)
plt.plot(co_num,co_stdlist,label='CV')
plt.xlabel('Sample number, s')
# plt.ylabel('Converage rate')
plt.ylim(0,0.2)
plt.legend()
plt.savefig('Fourmethod_std.jpg',dpi=600)
plt.show()


# In[1503]:


plt.plot(ponumlist,PO_stdlist,label='Mean area')
# plt.fill_between(ponumlist,PO_fill1list,PO_fill2list,color='#FF4500',alpha=0.1, edgecolor="white",label='Std')
plt.ylabel('Stander Deviation')
plt.xlim(1000,10000)
plt.xlabel('Sample number,s')
plt.legend()
plt.savefig('PO_change_std.jpg',dpi=600)
plt.rcParams['figure.figsize'] = (7.0, 5.0) 


# ### Convergence rate of each method

# In[1184]:


plt.rcParams['figure.figsize'] = (10.0, 7.0) 
plt.subplot(2,2,1)
plt.plot(itera_list,Coverageilist,label='FRS')
plt.xlabel('Iteration numbers, i')
plt.ylabel('Convergence rate')
plt.legend()
plt.subplot(2,2,2)
plt.plot(L_itera_list,L_i_Coverageilist,label='LHS')
plt.xlabel('Iteration numbers, i')
plt.ylabel('Convergence rate')
plt.legend()
plt.subplot(2,2,3)
plt.plot(o_i_maxiterlist,o_i_Coverageilist,label='OS')
plt.xlabel('Iteration numbers, i')
plt.ylabel('Convergence rate')
plt.legend()
plt.subplot(2,2,4)
plt.plot(i_co_maxiterlist,co_i_Coverageilist,label='CV')
plt.xlabel('Iteration numbers, i')
plt.ylabel('Convergence rate')
plt.legend()
plt.savefig('Fourmethod.jpg',dpi=600)
plt.show()


# In[1192]:


plt.plot(maxiterlist,np.abs(ia_arealist-True_Ais))
plt.xlabel('So_iteration number,i')
plt.ylabel('Convergence rate')
plt.savefig('Sobol_i_cov.jpg',dpi=600)


# In[1498]:


plt.plot(maxiterlist,np.abs(i_pd_arealist-True_Ais))
plt.xlabel('Iteration number,i')
plt.ylabel('Covergence rate')
plt.savefig('Pos_i_cov.jpg',dpi=600)


# In[ ]:





# In[ ]:




