#!/usr/bin/env python
# coding: utf-8

# ## Support Functions

# In[61]:


# generates a list of 0 with length n 
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

# generates a list of -1 with length n
def minusOneListMaker(n):
    listOfMinusOnes = [-1] * n
    return listOfMinusOnes

# deletes element out of a list
def delete(F0, index):
    NeueListe = []
    for i in range(len(F0)):
        if (F0[i] != index):
            NeueListe.append(F0[i])
    return NeueListe 


# ## Reading Data

# In[62]:


import numpy as np

A = []    # appearance time 
E = []    # earliest landing 
T = []    # target landing time 
L = []    # latest landing time
g = []    # penalty cost for landing early 
h = []    # penalty cost for landing late
S = []    # separation time between aircraft i and aircraft j (i lands before j)
Zdisp = 0       # displacement costs
lamb_disp = 1   # weight for displacement
lamb_cost = 1   # weight for total costs
p_disp = 1      # objective function weighting for unit of displacement 

# reading data 

with open("airland5.txt") as problem_file:
    data = problem_file.read()
    data = data.split()

    # number of planes
    P = int(data[0])
    # freeze time
    t_freeze = int(data[1])
    number_of_attributes = 6

    # save data to variables  
    for i in range(len(data)):
        if i in range(2, len(data), P + number_of_attributes):
            A.append(data[i])
        elif i in range(3, len(data), P + number_of_attributes):
            E.append(data[i])
        elif i in range(4, len(data), P + number_of_attributes):
            T.append(data[i])
        elif i in range(5, len(data), P + number_of_attributes):
            L.append(data[i])
        elif i in range(6, len(data), P + number_of_attributes):
            g.append(data[i])
        elif i in range(7, len(data), P + number_of_attributes):
            h.append(data[i])
        elif i in range(8, len(data), P + number_of_attributes):
            for j in range(P):
                S.append(data[i + j])
S = np.array(S)
shape = (P, P)
S = S.reshape(shape)

A = [int(i) for i in A]
E = [int(i) for i in E]
T = [int(i) for i in T]
L = [int(i) for i in L]
g = [float(i) for i in g]
h = [float(i) for i in h]
S = [list(map(int, x)) for x in S]


# ## Displacement ALP

# In[63]:


import gurobipy as gp
from gurobipy import GRB

def ALP_dynamic(Solution_List, F0, F1, F2, E, T, L, g, h, S):
    m = gp.Model("ALP_dynamic")    
    
    # calculate the number of planes P
    P = len(F1)
    
    # create F3 (which is F1 u F2)
    F3 = F1
    for i in range(len(F3)):
        if F2[i] != -1:
            F3[i] = F2[i]

    # VARIABLES
    time = m.addVars(P, name="time")
    delta = m.addVars(P, P, vtype=GRB.BINARY, name="delta")
    alpha = m.addVars(P, name="alpha")
    beta = m.addVars(P, name="beta")
    displacement_new_approach = m.addVars(P, name="displacement")
    helper = m.addVars(P, name="helper")
    
    # CONSTRAINTS
    
    # constraints for planes in F1
    m.addConstrs((time[i] >= E[i] for i in F1 if i != -1), name="Early")
    m.addConstrs((time[i] <= L[i] for i in F1 if i != -1), name="Late")
    m.addConstrs((delta[i, j] + delta[j, i] == 1 for i in F1 if i != -1 for j in F1 if i != -1 if j > i), 
                 name="Sequence")
    # constraint valid for planes in F1 & F2
    m.addConstrs((time[j] >= time[i] + S[i][j] * delta[i, j] - (L[i] - E[j]) * delta[j, i] 
                  for i in F3 if i != -1 for j in F3 if j != -1 if i != j), name = "Separation")
    # constraints just valid for F1
    m.addConstrs(alpha[i] >= T[i] - time[i] for i in F1 if i != -1)
    m.addConstrs(beta[i] >= time[i] - T[i] for i in F1 if i != -1)
    m.addConstrs(alpha[i] <= T[i] - E[i] for i in F1 if i != -1)
    m.addConstrs(beta[i] <= L[i] - T[i] for i in F1 if i != -1)
    m.addConstrs(beta[i] >= 0 for i in F1 if i != -1)
    m.addConstrs(alpha[i] >= 0 for i in F1 if i != -1)
    m.addConstrs(time[i] == T[i] - alpha[i] + beta[i] for i in F1 if i != -1)
    # planes in F2 have already assigned a time, which cannot be changed anymore
    for i in range(len(F2)):
        if F2[i] != -1:
            m.addConstr(time[i] == Solution_List[i])

    #displacement   
    for i in F1:
        if i != -1 and previous_solution[i] != 0:
            if previous_solution[i] < T[i]:
                m.addConstr(helper[i] >= 0)
                m.addConstr(helper[i] >= previous_solution[i] - time[i])
                m.addConstr(displacement_new_approach[i] == g[i] * helper[i])
            elif previous_solution[i] > T[i]:
                m.addConstr(helper[i] >= 0)
                m.addConstr(helper[i] >= time[i] - previous_solution[i])
                m.addConstr(displacement_new_approach[i] == h[i] * helper[i])    
            else:
                m.addConstr(helper[i] >= 0)
                m.addConstr(helper[i] >= g[i] * (previous_solution[i] - time[i]))         
                m.addConstr(helper[i] >= h[i] * (time[i] - previous_solution[i]))
                m.addConstr(displacement_new_approach[i] == helper[i])
                            
            
    # OBJECTIVE
    obj1 = lamb_cost * gp.quicksum(g[i] * alpha[i] + h[i] * beta[i] for i in range(P))
    obj2 = lamb_disp * p_disp * gp.quicksum(displacement_new_approach[i] for i in range(P)) 
    objective = obj1 + obj2
    m.ModelSense = gp.GRB.MINIMIZE    
    m.setObjective(objective)

    m.optimize()
    
    # RESULTS
    a = m.getAttr("x")
    z = m.objVal
    runtime = m.getAttr("Runtime") 
    
    landing_times = zerolistmaker(P)
    latest_landing = max(L)
    for i in range(P):
        if (a[i] <= latest_landing):
            landing_times[i] = a[i] 

    return landing_times, obj2.getValue(), obj1.getValue(), runtime


# ### First plane

# In[64]:


import numpy as np

previous_solution = zerolistmaker(P)
# set current time to min of all A
t = min(A)
index = A.index(t)
# set of aircrafts that have not yet appeared 
F0 = []
# set of aircrafts that have appeared but are not frozen
F1 = minusOneListMaker(P)
# set of aircrafts that have landed / are frozen
F2 = minusOneListMaker(P)

# fill F1
F1[index] = index
# M = sufficient large number 
A[index] = 100000
# fill F0
F0 = [x for x in list(range(P)) if x not in F1]

# solve ALP
R = ALP_dynamic(previous_solution, F0, F1, F2,
                                   list(E[i] for i in range(P)), 
                                   list(T[i] for i in range(P)), 
                                   list(L[i] for i in range(P)),
                                   list(g[i] for i in range(P)), 
                                   list(h[i] for i in range(P)),
                                   list(S[i] for i in range(P)))
# save solution
initial_solution, displacement, sol_cost, runtime  = R
# runtime
total_runtime = runtime
# save solution as previous solution for next iteration
previous_solution = initial_solution.copy()


# ### 2nd to Pth plane

# In[65]:


for m in range(P-1):

    # check, if there are still planes arriving:
    if (min(A) < 100000):  
        t = min(A)
        index = A.index(t)

        # update F1
        F1[index] = index
        # update A
        A[index] = 100000
        # update F0
        F0 = delete(F0, index)

        # F2 (after the update of F1)
        for i in range(P):
            if (float(t + t_freeze) >= previous_solution[i] and previous_solution[i] != 0.0):
                F2[i] = i
                F1[i] = -1

    # solve the displacement problem:
    R = ALP_dynamic(previous_solution, F0, F1, F2,
                                   list(E[i] for i in range(P)), 
                                   list(T[i] for i in range(P)), 
                                   list(L[i] for i in range(P)),
                                   list(g[i] for i in range(P)), 
                                   list(h[i] for i in range(P)),
                                   list(S[i] for i in range(P)))
    
    # save results
    initial_solution, displacement, sol_cost, runtime = R
    # add runtime
    total_runtime += runtime
    # add displacement costs
    Zdisp += displacement
    # update previous solution
    previous_solution = initial_solution.copy()


# In[66]:


# print results and runtime
print("Z_sol =", sol_cost)
print("Z_disp =", Zdisp)
total_costs = sol_cost + Zdisp
print("Total costs =", total_costs)
print("Total runtime(in seconds) =",total_runtime)


# In[ ]:




