#!/usr/bin/env python
# coding: utf-8

# ## Support Functions

# In[8]:


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

# In[9]:


import numpy as np

A = []    # appearance time 
E = []    # earliest landing 
T = []    # target landing time 
L = []    # latest landing time
g = []    # penalty cost for landing early 
h = []    # penalty cost for landing late
S = []    # separation time between aircraft i and aircraft j (i lands before j)

# file
instance = "airland5.txt"

# reading data

with open(instance) as problem_file:
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


# ## static ALP

# In[10]:


import gurobipy as gp
from gurobipy import GRB

def ALP(P, E, T, L, g, h, S):
    m = gp.Model("ALP_dynamic")    
    
    # VARIABLES
    time = m.addVars(P, name="time")
    delta = m.addVars(P, P, vtype=GRB.BINARY, name="delta")
    alpha = m.addVars(P, name="alpha")
    beta = m.addVars(P, name="beta")
    displacement_new_approach = m.addVars(P, name="displacement")
    helper = m.addVars(P, name="helper")

    # CONSTRAINTS
    # constraints for planes in F1
    m.addConstrs((time[i] >= E[i] for i in range(P)), name="Early")
    m.addConstrs((time[i] <= L[i] for i in range(P)), name="Late")
    m.addConstrs((delta[i, j] + delta[j, i] == 1 for i in range(P) for j in range(P) if j > i), 
                 name="Sequence")
    # constraint valid for planes in F1 & F2
    m.addConstrs((time[j] >= time[i] + S[i][j] * delta[i, j] - (L[i] - E[j]) * delta[j, i] 
                  for i in range(P) for j in range(P) if i != j), name = "Separation")
    # constraints just valid for F1
    m.addConstrs(alpha[i] >= T[i] - time[i] for i in range(P))
    m.addConstrs(beta[i] >= time[i] - T[i] for i in range(P))
    m.addConstrs(alpha[i] <= T[i] - E[i] for i in range(P))
    m.addConstrs(beta[i] <= L[i] - T[i] for i in range(P))
    m.addConstrs(beta[i] >= 0 for i in range(P))
    m.addConstrs(alpha[i] >= 0 for i in range(P))
    m.addConstrs(time[i] == T[i] - alpha[i] + beta[i] for i in range(P))
                
    
    # OBJECTIVE
    objective = gp.quicksum(g[i] * alpha[i] + h[i] * beta[i] for i in range(P)) 
    m.ModelSense = gp.GRB.MINIMIZE    
    m.setObjective(objective)

    m.optimize()
    
    # RESULTS
    a = m.getAttr("x")
    z = objective.getValue()
    runtime = m.getAttr("Runtime") 
    
    landing_times = zerolistmaker(P)
    latest_landing = max(L)
    for i in range(P):
        if (a[i] <= latest_landing):
            landing_times[i] = a[i] 

    return landing_times, z, runtime


# ## Run the model

# In[11]:


R = ALP(P, E, T, L, g, h, S)

landing_times, z, runtime = R

print("Total costs =", z)
print("Runtime =", runtime)


# In[ ]:




