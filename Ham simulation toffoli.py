#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pennylane import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt
import networkx as nx


# In[2]:


edges = [(0, 1), (1, 2)]
graph = nx.Graph(edges)

nx.draw(graph, with_labels=True)
plt.show()


# In[146]:

# code for Toffoli gate
U = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])

def U_actual():
    qml.QubitUnitary(U,wires=[0,1,2])
    
    
'''''''''''''

# In[4]:


# corresponding hamiltonian for toffoli gate using maximum two body interactions
l=np.random.rand(15)
H=qml.Hamiltonian([l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9],l[10],l[11],l[12],l[13],l[14]],
                                                    [qml.PauliX(wires=0)@qml.PauliX(wires=1),
                                                     qml.PauliX(wires=1)@qml.PauliX(wires=2),
                                                     qml.PauliY(wires=0)@qml.PauliY(wires=1),
                                                     qml.PauliY(wires=1)@qml.PauliY(wires=2),
                                                     qml.PauliZ(wires=0)@qml.PauliZ(wires=1),
                                                     qml.PauliZ(wires=1)@qml.PauliZ(wires=2),
                                                     qml.PauliZ(wires=0),qml.PauliZ(wires=1),qml.PauliZ(wires=2),
                                                     qml.PauliX(wires=0),qml.PauliX(wires=1),qml.PauliX(wires=2),
                                                     qml.PauliY(wires=0),qml.PauliY(wires=1),qml.PauliY(wires=2)])
print(H)

''''''''''''

# In[147]:

# this part of code will produce the unitary transformation of the corresponding hamiltonian given in the prev cell 
coeffs = [1]
obs = [
    qml.PauliZ(0) @ qml.PauliZ(1)@qml.Identity(2)
]
H_toy = qml.vqe.Hamiltonian(coeffs, obs)
print(H_toy)


# In[148]:


dev = qml.device('default.qubit', wires=3)
t = 1
n = 1

@qml.qnode(dev)
def circuit3():
    qml.templates.ApproxTimeEvolution(H_toy, t, n)
    U_actual()
    return qml.state() # represents the output state
print(circuit3())
print(circuit3().shape)
print(circuit3.draw())

#circuit for representing exp(-iHt), for any local Ham H

'''''''''''''
# In[161]:


dev = qml.device('default.qubit', wires=3)


#@qml.qnode(dev)
def ansatz(param,**kwargs):
    l=param
   # wires=l[0]
    #[qml.PauliX(wires=i) for i in range(3)]
    #[qml.Hadamard(wires=i) for i in range(3)]
    
    # two qubit gates    
    # circuit for local ham H=Z⊗Z⊗I    
    qml.CNOT(wires=[0,1])
    qml.RZ(l[0],wires=1)
    qml.CNOT(wires=[0,1])
    #qml.Identity(wires=2)
    # circuit for local ham H=I⊗Z⊗Z
    qml.CNOT(wires=[1,2])
    qml.RZ(l[1],wires=2)
    qml.CNOT(wires=[1,2])
    #qml.Identity(wires=0)
    # circuit for local ham H=I⊗X⊗X
    qml.CNOT(wires=[1,2])
    qml.RX(l[2],wires=2)
    qml.CNOT(wires=[1,2])
    #qml.Identity(wires=0)
    # circuit for local ham H=X⊗X⊗I    
    qml.CNOT(wires=[0,1])
    qml.RX(l[3],wires=1)
    qml.CNOT(wires=[0,1])
    #qml.Identity(wires=3)   
    # circuit for local ham H=Y⊗Y⊗I    
    qml.CNOT(wires=[0,1])
    qml.RY(l[4],wires=1)
    qml.CNOT(wires=[0,1])
    #qml.Identity(wires=3)   
    # circuit for local ham H=I⊗Y⊗Y
    qml.CNOT(wires=[1,2])
    qml.RY(l[5],wires=2)
    qml.CNOT(wires=[1,2])
    #qml.Identity(wires=0)
    
    # single qubit gates
    
    # circuit for local ham H=I⊗I⊗Z+I⊗I⊗X+I⊗I⊗Y
    #[qml.Identity(wires=i) for i in range(2)]
    qml.RZ(l[6],wires=2)
    #[qml.Identity(wires=i) for i in range(2)]
    qml.RX(l[7],wires=2)
    #[qml.Identity(wires=i) for i in range(2)]
    qml.RY(l[8],wires=2)
    # circuit for local ham H=I⊗Z⊗I+I⊗X⊗I+I⊗Y⊗I
    #qml.Identity(wires=0) 
    qml.RZ(l[9],wires=1)
    #qml.Identity(wires=2)
    #qml.Identity(wires=0) 
    qml.RX(l[10],wires=1)
    #qml.Identity(wires=2)
    #qml.Identity(wires=0) 
    qml.RY(l[11],wires=1)
    #qml.Identity(wires=2)
    # circuit for local ham H=Z⊗I⊗I+X⊗I⊗I+Y⊗I⊗I
    qml.RZ(l[12],wires=0)
   # [qml.Identity(wires=i) for i in range(1,3)]
    qml.RX(l[13],wires=0)
   # [qml.Identity(wires=i) for i in range(1,3)]
    qml.RY(l[14],wires=0)
   # [qml.Identity(wires=i) for i in range(1,3)]
    #[qml.Hadamard(wires=i) for i in range(3)]
    #return [qml.expval(qml.PauliZ(i)) for i in range(3)]
    #return qml.expval(qml.PauliZ(0))
    


# ![ham.png](attachment:ham.png)

# In[162]:


dev = qml.device('default.qubit', wires=3)
#@qml.qnode(dev) [it should be used when we want to return something from a function]

def circuit(params,**kwargs):
    
    #wires=params[0]
    #qml.layer(ansatz,3,params)
    #[qml.Hadamard(wires=i) for i in range(3)]
    [qml.PauliX(wires=i) for i in range(3)]
    ansatz(params)
    U_actual()
    
    
    #[qml.PauliX(wires=i) for i in range(3)]
    #return [qml.expval(qml.PauliZ(i)) for i in range(3)]


# In[12]:


b=np.array([np.random.rand(45)])
print(b)


# In[163]:


obs = [qml.PauliZ(0),qml.PauliZ(1) , qml.PauliZ(2)]
H = qml.vqe.Hamiltonian([1,1,1], obs)
print(H)
dev = qml.device("default.qubit", wires=3)

######################
#ansatz = qml.templates.StronglyEntanglingLayers
#def ansatz(weights,wires,**kwargs):
    #qml.templates.StronglyEntanglingLayers(weights=weights,wires=wires)
    #qml.templates.ApproxTimeEvolution(H3,t,n)
    #U_actual
    #[qml.PauliX(wires=i) for i in range(3)]
######################    

cost_opt = qml.ExpvalCost(circuit, H, dev, optimize=True)
cost_no_opt = qml.ExpvalCost(circuit, H, dev, optimize=False)

initparam= np.random.rand(15)
params = initparam


# In[164]:


cost_history = []
steps=500
for it in range(steps):
    params, cost = qml.GradientDescentOptimizer(stepsize=0.01).step_and_cost(cost_opt, params)
    #print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    if (it+1)%10==0:
        
        print("the value of cost at step {:} is {:.5f}".format(it+1,cost_opt(params)))

        cost_history.append(cost)
print(cost)    


# In[165]:


x=np.linspace(0,500,50)
y=cost_history
plt.plot(x,y,"o-")
plt.xlabel("No. of steps")
plt.ylabel("Cost")
plt.title("Variational cost Vs Time step")
#plt.grid()
plt.show()


# In[177]:


params==initparam


# In[178]:


print(initparam)
print(params)


# In[23]:


ansatz(params)
print(ansatz.draw())

# the final circuit and the corresponding parameters of the ising hamiltonian
# we can get the circuit diag as output when we return something in the fucntion e.g. qml.expval(qml.Pauli(i)....)


# In[81]:


dev = qml.device('default.qubit', wires=3)
# for playing with the ansatz we can use myansatz anytime
def myansatz(param,**kwargs):
    l=param  
    [qml.Hadamard(wires=i) for i in range(3)]
    # circuit for local ham H=Z⊗Z⊗I    
    qml.CNOT(wires=[0,1])
    qml.RZ(l[0],wires=1)
    qml.CNOT(wires=[0,1])
    #qml.Identity(wires=2)
    # circuit for local ham H=I⊗Z⊗Z
    qml.CNOT(wires=[1,2])
    qml.RZ(l[1],wires=2)
    qml.CNOT(wires=[1,2])
    #qml.Identity(wires=0)
    # circuit for local ham H=I⊗X⊗X
    qml.CNOT(wires=[1,2])
    qml.RX(l[2],wires=2)
    qml.CNOT(wires=[1,2])
    #qml.Identity(wires=0)
    # circuit for local ham H=X⊗X⊗I    
    qml.CNOT(wires=[0,1])
    qml.RX(l[3],wires=1)
    qml.CNOT(wires=[0,1])
    #qml.Identity(wires=3)   
    # circuit for local ham H=Y⊗Y⊗I    
    qml.CNOT(wires=[0,1])
    qml.RY(l[4],wires=1)
    qml.CNOT(wires=[0,1])
    #qml.Identity(wires=3)   
    # circuit for local ham H=I⊗Y⊗Y
    qml.CNOT(wires=[1,2])
    qml.RY(l[5],wires=2)
    qml.CNOT(wires=[1,2])
    #qml.Identity(wires=0)
    
    # single qubit gates
    
    # circuit for local ham H=I⊗I⊗Z+I⊗I⊗X+I⊗I⊗Y
    #[qml.Identity(wires=i) for i in range(2)]
    qml.RZ(l[6],wires=2)
    #[qml.Identity(wires=i) for i in range(2)]
    qml.RX(l[7],wires=2)
    #[qml.Identity(wires=i) for i in range(2)]
    qml.RY(l[8],wires=2)
    # circuit for local ham H=I⊗Z⊗I+I⊗X⊗I+I⊗Y⊗I
    #qml.Identity(wires=0) 
    qml.RZ(l[9],wires=1)
    #qml.Identity(wires=2)
    #qml.Identity(wires=0) 
    qml.RX(l[10],wires=1)
    #qml.Identity(wires=2)
    #qml.Identity(wires=0) 
    qml.RY(l[11],wires=1)
    #qml.Identity(wires=2)
    # circuit for local ham H=Z⊗I⊗I+X⊗I⊗I+Y⊗I⊗I
    qml.RZ(l[12],wires=0)
   # [qml.Identity(wires=i) for i in range(1,3)]
    qml.RX(l[13],wires=0)
   # [qml.Identity(wires=i) for i in range(1,3)]
    qml.RY(l[14],wires=0)
   # [qml.Identity(wires=i) for i in range(1,3)]


# In[174]:


dev = qml.device('default.qubit', wires=3)
#@qml.qnode(dev) [it should be used when we want to return something from a function]

# params are the optimal ones
def circ(params,**kwargs):
    
    #wires=params[0]
    #qml.layer(ansatz,3,params)
    ansatz(params)
    U_actual()
    
    
    #[qml.PauliX(wires=i) for i in range(3)]
    #return [qml.expval(qml.PauliZ(i)) for i in range(3)]


# In[175]:



@qml.qnode(dev)
def probability_circuit(params):
    [qml.PauliX(wires=i) for i in range(3)]
    circ(params)
    return qml.probs(wires=[0,1,2])


probs = probability_circuit(params)
#print(len(probs))


# In[176]:


plt.style.use("seaborn")
plt.bar(range(2 ** 3), probs)
plt.show()


# In[42]:


#print(np.linspace(0,500,500))


# In[152]:


# for understanding how the toffoli gate works
@qml.qnode(dev)
def prob_circuit(params):
    [qml.PauliX(wires=i) for i in range(2)]
    #circ(params)
    U_actual()
    return qml.probs(wires=[0,1,2])


prob = prob_circuit(params)
#print(len(probs))


# In[153]:


plt.style.use("seaborn")
plt.bar(range(2 ** 3), prob)
plt.show()


# In[154]:


print(prob_circuit.draw())


# In[ ]:




