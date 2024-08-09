#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def f(x):
    return 3*x**2 - 4*x + 5


# In[3]:


f(3.0)


# In[4]:


xs = np.arange(-5 , 5 , 0.25)
ys = f(xs)
ys
plt.plot(xs , ys)


# In[5]:


h = 0.000000001
# x = -3.0
# x = 3.0

x = 2/3 # slope 0

# differentiator 
# This is how much the function responded in +ve direction plus i have normalised it as well by dividing it by h
# slope
(f(x+h) - f(x))/h


# In[6]:


# Les get more complex 
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)


# In[7]:


h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0
d = a*b + c

# d1 = a*b + c
# # wrt a
# a += h 
# d2 = a*b + c

# d1 = a*b + c
# # wrt b
# b += h 
# d2 = a*b + c

d1 = a*b + c
# wrt c
c += h 
d2 = a*b + c

print('d1' , d1)
print('d2' , d2)

print('slope' , (d2-d1)/h)


# In[8]:


# Value Object
class Value:
    
    def __init__(self , data , _children=()):
        self.data = data
        self._prev = set(_children)
     
    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self , other))
        return out;
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self , other))
        return out;

# repr helping to prinout data better
# a+b # a.__add__(b) here self = a and b = other
# a*b + c # (a.__mul__(b)).__add__(c)
# _children() empty tupple for efficiency
    
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d =  a*b + c
d


# In[9]:


d._prev


# In[10]:


# Value Object here _op is for which operation created those values
class Value:
    
    def __init__(self , data , _children=() , _op = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
     
    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self , other) , '+')
        return out;
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self , other) , '*')
        return out;
    
    
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d =  a*b + c
d


# In[11]:


d._prev


# In[12]:


d._op


# In[13]:


from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ data %.4f }" % (n.data,), shape='record')
        
        # we created these fake nodes just for visilaize like this fake node contains (+) or (*)
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
        
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


# In[14]:


draw_dot(d)


# In[15]:


# Adding Labels
class Value:
    
    def __init__(self , data , _children=() , _op = '', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
     
    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self , other) , '+')
        return out;
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self , other) , '*')
        return out;
    
    
a = Value(2.0 , label='a')
b = Value(-3.0 , label = 'b')
c = Value(10.0 , label = 'c')
e = a*b ; e.label = 'e'
d = e+c; d.label = 'd'
d =  a*b + c


from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label , n.data), shape='record')
        
        # we created these fake nodes just for visilaize like this fake node contains (+) or (*)
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
        
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


# In[16]:


draw_dot(d)


# In[17]:


# creating new value object at d
# Adding Labels
class Value:
    
    def __init__(self , data , _children=() , _op = '', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
     
    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self , other) , '+')
        return out;
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self , other) , '*')
        return out;
    
    
a = Value(2.0 , label='a')
b = Value(-3.0 , label = 'b')
c = Value(10.0 , label = 'c')
e = a*b ; e.label = 'e'
d = e+c; d.label = 'd'
f = Value(-2.0 , label='f')
L = d*f ; L.label ='L'
L


# In[18]:


draw_dot(L)


# # Back Propagation 

# * L Derivative w.r.t. L and L wrt f and L wrt d and wrt to c , e , a, b
# 
# * Next we are going to create variable inside the value class that maintains the derivative of L w.r.t. to that value and we wil call this variable grad

# In[19]:


class Value:
    
    def __init__(self , data , _children=() , _op = '', label=''):
        self.data = data
        self.grad = 0.0 # 0 means no effect initially it donot effect
        self._prev = set(_children)
        self._op = _op
        self.label = label
     
    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self , other) , '+')
        return out;
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self , other) , '*')
        return out;
    
    
a = Value(2.0 , label='a')
b = Value(-3.0 , label = 'b')
c = Value(10.0 , label = 'c')
e = a*b ; e.label = 'e'
d = e+c; d.label = 'd'
f = Value(-2.0 , label='f')
L = d*f ; L.label ='L'
L


# In[20]:


from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label , n.data , n.grad), shape='record')
        
        # we created these fake nodes just for visilaize like this fake node contains (+) or (*)
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
        
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


# In[21]:


draw_dot(L)


# In[22]:


def lol():
    h = 0.0001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0 + h, label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# In[23]:


# changing L by h
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L2 = L.data + h
    
    print((L2-L1) / h)
    
lol()


# In[24]:


# filling back propagation manually 
L.grad = 1.0


# In[25]:


draw_dot(L)


# In[26]:


# L = d*f
# # dL / dd = ? it is f
# (f(x+h)-f(x))/h

# ((d+h)*f - d_f)/h
# (d*f + h*f - d*f) / h
# (h*f)/h
# f


# In[27]:


f.grad = 4.0 # is just the value of d
d.grad = -2.0


# In[28]:


draw_dot(L)


# In[29]:


# Now checking whether it's correct or not
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 + h, label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# In[30]:


# Now checking whether it's correct or not
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    d.data += h
    f = Value(-2.0, label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# In[31]:


Now dL / dc and dL/de

first we calculate dd/dc ? 1.0

d = c+e
f(x+h) - f(x) / h 
((c+e + e) - (c+e))/h
c+h+e+c-e /h ? 1.0

dd/dc = 1.0
by symmetry dd/de = 1.0

WANT:
dL/dc 

KNOW:
dL/dd
dd/dc (1.0)

Using chain rule :
dL/dc = (dL/dd) * (dd/dc)
dd/dc (1.0) 
so :
dL/dc = (dL/dd) 


# In[32]:


c.grad = -2.0
e.grad = -2.0


# In[33]:


draw_dot(L)


# In[34]:


# Verifying
# Now checking whether it's correct or not
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    c.data += h
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# In[35]:


# Verifying
# Now checking whether it's correct or not
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    e.data += h
    d = e+c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# In[36]:


# For 1st Node
dL / de = -2.0
dL / da = ?

e = a*b # differentiating wrt to a we get b
de / da ? b(-3.0)
de / da ? a(-2.0)
dL / da = (dL / de) * (de / da)


# In[37]:


a.grad = (-2.0 * -3.0)
b.grad = (-2.0 * -2.0)


# In[38]:


draw_dot(L)


# In[39]:


# Verifying
# Now checking whether it's correct or not
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    a.data += h
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# In[40]:


# Verifying
# Now checking whether it's correct or not
def lol():
    h = 0.001
    
    a = Value(2.0 , label='a')
    b = Value(-3.0 , label = 'b')
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f ; L.label ='L'
    L1 = L.data
    
    a = Value(2.0, label='a')
    b = Value(-3.0 , label = 'b')
    b.data += h
    c = Value(10.0 , label = 'c')
    e = a*b ; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f ; L.label ='L'
    L2 = L.data
    
    print((L2-L1) / h)
    
lol()


# ### Let's see power in action : we are going to nudge our input to try to make L go up

# In[41]:


# if you want L to go up we jsut want it in direction of gradient
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
d.data += 0.01 * d.grad

e = a*b 
d = e+c
L = d*f 

print(L.data)
# earlier it was -8.0


# In[42]:


plt.plot(np.arange(-5,5,0.2) , np.tanh(np.arange(-5,5,0.2))); plt.grid();


# In[43]:


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.7, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

draw_dot(n)


# In[44]:


# Implementing exponentiation
class Value:
    
    def __init__(self , data , _children=() , _op = '', label=''):
        self.data = data
        self.grad = 0.0 # 0 means no effect initially it donot effect
        self._prev = set(_children)
        self._op = _op
        self.label = label
     
    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self , other) , '+')
        return out;
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self , other) , '*')
        return out;
    
#     for tanh
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out
        
    
    
a = Value(2.0 , label='a')
b = Value(-3.0 , label = 'b')
c = Value(10.0 , label = 'c')
e = a*b ; e.label = 'e'
d = e+c; d.label = 'd'
f = Value(-2.0 , label='f')
L = d*f ; L.label ='L'
L


# In[45]:


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.7, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

draw_dot(o)


# In[46]:


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

draw_dot(o)


# # Now doing back propagation

# In[47]:


o.grad = 1.0


# In[48]:


draw_dot(o)


# In[49]:


# to back propagate through tanh we need to know the local derivative of tanh 
o = tanh(n)
do / dn = 1-tanh(n)**2
do / dn = 1-o**2


# In[50]:


1-o.data**2


# In[51]:


n.grad = 0.5


# In[52]:


draw_dot(o)


# In[53]:


# Now the local derivative is '+' so it flow simply into them


# In[54]:


x1w1x2w2.grad = 0.5
b.grad = 0.5


# In[55]:


draw_dot(o)


# In[56]:


x1w1.grad = 0.5
x2w2.grad = 0.5
draw_dot(o)


# In[57]:


x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

draw_dot(o)

# w1 grad is 1 so if you want output(o) to increase you need to increase w1 becuase it's grad = 1 which means directly proportional


# # Backward Propagation Automatically

# In[58]:


# # Implementing backward
# class Value:
    
#     def __init__(self , data , _children=() , _op = '', label=''):
#         self.data = data
#         self.grad = 0.0 # 0 means no effect initially it donot effect
#         # we are going to store how are we going to change the output gradient into the inputs gradients
#         # by default this function doesnot do anything e.g. for leaf node
#         self.backward = lambda : None
#         self._prev = set(_children)
#         self._op = _op
#         self.label = label
     
    
    
    
    
    
#     def __repr__(self):
#         return f"Value(data = {self.data})"
    
    
    
    
    
    
#     def __add__(self, other):
#         out = Value(self.data + other.data , (self , other) , '+')
        
#         # Our job is to take out's grad and propogate it inot self's grad and other.grad
#         def _backward():
#             self.grad += 1.0 * out.grad
#             other.grad += 1.0 * out.grad
#             # Out's grad will simply be copied onto sel'f grad and other's grad for addition operation
            
        
#         # out's backward should be the 
# #         this dnot return none so _backward() else simple _backward
#         out._backward = _backward
#         return out;
    
    
    
    
    
#     def __mul__(self, other):
#         out = Value(self.data * other.data , (self , other) , '*')
#         def _backward():
#             self.grad += (other.data * out.grad)
#             other.grad += self.data * out.grad
            
#         out._backward = _backward
#         return out;
    
    
    
    
    
    
    
# #     for tanh
#     def tanh(self):
#         x = self.data
#         t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
#         out = Value(t, (self, ), 'tanh')
        
#         def _backward():
#             self.grad += (1-t**2) * out.grad
        
#         out._backward = _backward
#         return out
        
    
    
# a = Value(2.0 , label='a')
# b = Value(-3.0 , label = 'b')
# c = Value(10.0 , label = 'c')
# e = a*b ; e.label = 'e'
# d = e+c; d.label = 'd'
# f = Value(-2.0 , label='f')
# L = d*f ; L.label ='L'
# L


# In[59]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward

        return out
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward

        return out


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[60]:


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

draw_dot(o)


# In[61]:


# Automatic Propagation
# Base case
o.grad = 1.0
draw_dot(o)


# In[62]:


o._backward()
draw_dot(o)


# In[63]:


n._backward()
draw_dot(o)


# In[64]:


b._backward()
draw_dot(o)


# In[65]:


x1w1x2w2._backward()
draw_dot(o)


# In[66]:


x2w2._backward()
x1w1._backward()

draw_dot(o)


# ## Implementing automatic backward propogation for whole graph

# In[67]:


# ordering of graph -> Topological sort , all edges from left to right
# TOPOLOGICAL GRAPH

# starts at o if not visited marks it as visited and then iterates throigh all the childrens and calls build topological on them 
# o add its self to list after all children have been processed
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)
topo


# In[68]:


# resseting gradients
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

draw_dot(o)


# In[69]:


# we will call ._backward on topological order
o.grad = 1.0
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)

for node in reversed(topo):
    node._backward()


# In[70]:


draw_dot(o)


# In[71]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
      
        return out
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[72]:


# resseting gradients
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

draw_dot(o)


# In[73]:


o.backward()


# In[74]:


draw_dot(o)


# # Fixing a BUG 
# when one node is used multiple times

# In[75]:


a = Value(3.0 , label='a')
b = a+a ; b.label = 'b'
b.backward()
draw_dot(b)


# In[76]:


a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b    ; d.label = 'd'
e = a + b    ; e.label = 'e'
f = d * e    ; f.label = 'f'

f.backward()

draw_dot(f)


# In[77]:


# We were overiding earlier now we are incrementing


# In[78]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[79]:


a = Value(3.0 , label='a')
b = a+a ; b.label = 'b'
b.backward()
draw_dot(b)
# now it's correct


# In[80]:


a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b    ; d.label = 'd'
e = a + b    ; e.label = 'e'
f = d * e    ; f.label = 'f'

f.backward()

draw_dot(f)


# # Breaking up tanh, exercising with more operations

# In[81]:


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

o.backward()
draw_dot(o)


# In[82]:


# this will give error
a = Value(2.0)
a + 1


# In[ ]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[ ]:


a = Value(2.0)
a + 1


# In[83]:


# same for multiple 
# this will give error
a = Value(2.0)
a * 2


# In[84]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[85]:


a = Value(2.0)
a * 2 # a.__mul__(2)


# In[86]:


# will give error
2*a # 2.__mul__(a)


# In[87]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
    
#     swapping operands
    def __rmul__(self,other):
        return self*other
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[88]:


2*a


# In[89]:


# for exponentiaion
import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
    
#     swapping operands
    def __rmul__(self,other):
        return self*other
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            # out.data is local derivative here
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward

        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[90]:


a.exp()


# In[91]:


# Division
a = Value(2.0)
b = Value(4.0)

a/b


# In[92]:


# same thing 
a/b
a * (1/b)
a * (b**-1)



# In[93]:


# for exponentiaion
import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

#     swapping operands
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            # out.data is local derivative here
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward

        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[94]:


# Division so the forward pass works
a = Value(2.0)
b = Value(4.0)

a/b


# In[260]:


# for exponentiaion
import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

#     swapping operands
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            # out.data is local derivative here
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward

        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


# In[96]:


# Does the backward pass also works? but before that we will see whether - works
a = Value(2.0)
b = Value(4.0)
a-b


# In[261]:


# resseting gradients
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()

draw_dot(o)


# In[98]:


# Both Forward(data) and Backward PaSs(gradient) are working and correct
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron

# as we increase the bias it moves towards 8
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# Changing defination of o
# ---------------------------- #
e = (2*n).exp()
o = (e - 1) / (e + 1) 
# ---------------------------- #

o.label = 'o'
o.backward()

draw_dot(o)


# In[99]:


2


# # Doing the same thing with PYTORCH

# In[100]:


import torch


# In[101]:


torch.tensor([[1,2,3],[3,4,5]])


# In[102]:


torch.tensor([[2.0]]).dtype


# In[103]:


# we convert it to double


# In[104]:


x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())


# In[105]:


o


# In[106]:


o.item()


# In[107]:


x2.grad


# # Building our own Neural Network

# In[108]:


import random
class Neuron:
  
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1)) # bias
  
    def __call__(self, x):
        # w * x + b
        return 0.0

x = [2.0 , 3.0]
n = Neuron(2)
n(x)


# In[109]:


# forward pass of this neuron


# In[110]:


import random
class Neuron:
  
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1)) # bias
  
    def __call__(self, x):
        # w * x + b
        print(list(zip(self.w , x)))
        return 0.0

x = [2.0 , 3.0]
n = Neuron(2)
n(x)


# In[111]:


# different output each time
class Neuron:
  
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1)) # bias
  
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

x = [2.0 , 3.0]
n = Neuron(2)
n(x)


# In[112]:


class Neuron:
  
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1)) # bias
  
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
  
    def parameters(self):
        return self.w + [self.b]

class Layer:
  
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
  
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    
x = [2.0 , 3.0]
n = Layer(2,3)
n(x)


# In[113]:


class Neuron:
  
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1)) # bias
  
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
  
    def parameters(self):
        return self.w + [self.b]

class Layer:
  
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
  
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
  
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
x = [2.0 , 3.0 , -1.0]
n = MLP(3 , [4,4,1]) # we want 3 inputs into 2 layers of 4 and 1 output
n(x)


# In[114]:


class Neuron:
  
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1)) # bias
  
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
  
    def parameters(self):
        return self.w + [self.b]

class Layer:
  
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
  
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
  
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
x = [2.0 , 3.0 , -1.0]
n = MLP(3 , [4,4,1]) # we want 3 inputs into 2 layers of 4 and 1 output
n(x)


# In[115]:


draw_dot(n(x))


# In[116]:


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
  
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
  
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# In[117]:


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)


# ## Creating a Tiny Dataset Writing the Loss Function

# In[118]:


# BackPropagation

xs = [
  [Value(2.0), Value(3.0), Value(-1.0)],
  [Value(3.0), Value(-1.0), Value(0.5)],
  [Value(0.5), Value(1.0), Value(1.0)],
  [Value(1.0), Value(1.0), Value(-1.0)],
]
ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)] # desired targets
ypred = [n(x) for x in xs]
ypred


# In[119]:


loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
print(f"Loss: {loss.data}")


# In[120]:


# Now we want to minimze the loss


# In[121]:


loss.backward()


# In[122]:


n.layers[0].neurons[0].w[0]


# In[123]:


# Now this value also has a grad due to backward pass
# influence on loss is also -ve, so increasing the weight of this particular layer will make the loss go down
n.layers[0].neurons[0].w[0].grad


# In[124]:


draw_dot(loss)


# # Collecting all the parameters of Neural Network
# We need some conveniniece code to gather up on all of the neural net so that we can operate on all of them simultanesously and every one of them we will nudge a tiny amount based on the gradient info.

# Just implemented again nothing's changed 

# In[332]:


import math

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
  
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
    
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward

        return out
  
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()


# In[333]:


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# In[334]:


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)


# In[335]:


n.parameters()


# In[336]:


print(len(n.parameters()))


# # Gradient Descent Optimization Manually

# In[337]:


# BackPropagation
xs = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]
ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)] # desired targets
ypred = [n(x) for x in xs]


# In[338]:


loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
print(f"Loss: {loss.data}")


# In[339]:


loss.backward()


# In[340]:


n.layers[0].neurons[0].w[0].data


# In[341]:


n.layers[0].neurons[0].w[0].grad


# In[342]:


# small effect in gradient descent scheme, we are thinking of a gradient poiting in the direction of increased loss
# in gradient descent we are modifying p.data by a smell step size


# In[343]:


# if data goes lower it will increase the gradient loss becuase deravitaive of neuron is -ve
# so will add -0.01 to it -> becuase we want to minimize the loss not maximize the loss
for p in n.parameters():
    p.data += -0.01 * p.grad


# In[344]:


n.layers[0].neurons[0].w[0].data


# In[345]:


n.layers[0].neurons[0].w[0].grad


# In[346]:


ypred = [n(x) for x in xs]
loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
print(f"Loss: {loss.data}")


# In[404]:


# Forward PASS
ypred = [n(x) for x in xs]
loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
print(f"Loss: {loss.data}")


# In[405]:


# USING BACKWARD PASSES
loss.backward()


# In[406]:


# UPDATE
for p in n.parameters():
    p.data += -0.01 * p.grad


# In[407]:


ypred
# Now after multiplease training of forward and backward passes now these y_preds have moved considerably high towards their target


# In[409]:


n.parameters()


# In[415]:


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)


# In[416]:


# BackPropagation
xs = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]
ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)] # desired targets


# In[417]:


for k in range(20):
  
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
    # Backward pass
    for p in n.parameters():
#         make zero grad just like it's in the constructor make it zero grad before backward pass
        p.grad = 0.0
    loss.backward()
  
    # Update
    for p in n.parameters():
        p.data += -0.1 * p.grad
  
    print(k, loss.data)


# In[419]:


ypred


# In[ ]:




