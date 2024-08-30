import pandas as pd
import numpy as np
import shin
from scipy.stats import beta
from scipy.optimize import minimize, NonlinearConstraint
import itertools
from tqdm import tqdm

def shin_standardisation(list_odds):
    return shin.calculate_implied_probabilities(list_odds)

def g_sigma(b1,b2):
    return 1+(b1*b2-1)/(2+b1+b2)

def rule_value(p,b1,b2):
    e1=p*(1+b1)-1
    e2=(1-p)*(1+b2)-1
    return p*(e1-e2)+e2*(1+e1)

def g_sigma_range(p_range, margin_range,e1_range,e2_range):
    g = []
    combs = list(itertools.product(p_range, margin_range, e1_range))
    for p,m,e1 in combs:
        b1 = np.clip((e1+1-p)/p,0,None)
        b2 = np.clip((1+b1)/(m*(1+b1)-1)-1,(min(e2_range)+p)/(1-p),(max(e2_range)+p)/(1-p))
        g.append(g_sigma(b1,b2))
    return [min(g),max(g)]

def gen_constraints(p_range,margin_range,e1_range,e2_range,rule_range):
    p_constraint = NonlinearConstraint(lambda x:x[0],p_range[0],p_range[1])
    margin_constraint = NonlinearConstraint(lambda x:1/(1+x[1])+1/(1+x[2]),margin_range[0],margin_range[1])
    e1_constraint = NonlinearConstraint(lambda x:x[0]*(1+x[1])-1,e1_range[0],e1_range[1])
    e2_constraint = NonlinearConstraint(lambda x:(1-x[0])*(1+x[2])-1,min(e2_range),max(e2_range))
    rule_constraint = NonlinearConstraint(lambda x:rule_value(x[0],x[1],x[2]),min(rule_range),max(rule_range))
    return [p_constraint,margin_constraint,e1_constraint,e2_constraint,rule_constraint]

#This objective was the most consistent in producing values which fulfilled the range criteria, while also giving a large range and variety amongst the (p,b1,b2).
#The distributions are by no means uniform, but the main point is a level playing field for comparing F* and Fsigma.
def obj(x,tgt,margin_range,e1_range,e2_range):
    return 6*(g_sigma(x[1],x[2])-tgt)**2 + (1/(1+x[1])+1/(1+x[2])-np.mean(margin_range))**2 + (x[0]*(1+x[1])-1-np.mean(e1_range))**2 + ((1-x[0])*(1+x[2])-1-np.mean(e2_range))**2

def gen_pbb(p_range,g_range,constraints,margin_range, e1_range,e2_range):
    tgt = np.random.uniform(g_range[0],g_range[1],1)[0]
    return minimize(obj,args=(tgt,margin_range,e1_range,e2_range),x0=[np.random.uniform(i[0],i[1],1)[0] for i in [p_range,(0,10),(0,10)]],bounds=[p_range,(0,100),(0,100)],constraints=constraints).x

p_range = [0.05,0.95]
margin_range = [0.95,1]
e1_range = [0,0.05]
rule_range = [0.01,1000]

its = 2000000
lpositive = []
e2_range = [0,0.05]
g_range = g_sigma_range(p_range, margin_range,e1_range,e2_range)
constraints = gen_constraints(p_range,margin_range,e1_range,e2_range,rule_range)
for i in tqdm(range(its)):
    lpositive.append(gen_pbb(p_range,g_range,constraints,margin_range,e1_range,e2_range))

lnegative=[]
e2_range = [-0.05,0]
g_range = g_sigma_range(p_range, margin_range,e1_range,e2_range)
constraints = gen_constraints(p_range,margin_range,e1_range,e2_range,rule_range)
for i in tqdm(range(its)):
    lnegative.append(gen_pbb(p_range,g_range,constraints,margin_range,e1_range,e2_range))

df_positive = pd.DataFrame(lpositive,columns=['p','b1','b2'])
df_negative = pd.DataFrame(lnegative,columns=['p','b1','b2'])

df_positive['p_est'] = df_positive[['b1','b2']].apply(lambda x: shin_standardisation(x+1+1e-5)[0],axis=1)
df_negative['p_est'] = df_negative[['b1','b2']].apply(lambda x: shin_standardisation(x+1+1e-5)[0],axis=1)

df_positive['nu'] = 99*np.sqrt(1-4*(df_positive['p']-0.5)**2)
df_positive['alpha'] = df_positive['p']*df_positive['nu']
df_positive['beta'] = (1-df_positive['p'])*df_positive['nu']
df_negative['nu'] = 99*np.sqrt(1-4*(df_negative['p']-0.5)**2)
df_negative['alpha'] = df_negative['p']*df_negative['nu']
df_negative['beta'] = (1-df_negative['p'])*df_negative['nu']
df_positive = df_positive.drop(columns='nu')
df_negative = df_negative.drop(columns='nu')

df_positive.to_csv('positive.csv',index=None)
df_negative.to_csv('negative.csv',index=None)

df_positive = pd.read_csv('positive.csv',index_col=None)
df_negative = pd.read_csv('negative.csv',index_col=None)


ns = [10000,100000,500000]
ws = [1e10,1e100,1e250]
p_case = 0
e2_range = [0,0.05]
if e2_range == [0,0.05]:
    df = df_positive.copy()
else:
    df = df_negative.copy()
if p_case == 1:
    df['p_use'] = df['p_est']
else:
    df['p_use'] = df['p']
l = []
for i in tqdm(range(10000-len(l))):
    dfi = df.sample(n=ns[-1],replace=True).reset_index(drop=True)
    w_sigma = np.cumsum(np.log(1+(dfi['b1']*dfi['b2']-1)/(2+dfi['b1']+dfi['b2'])))
    w_opt = np.cumsum(np.log(np.where(np.random.random(size=ns[-1])<(dfi['p'] if p_case != 2 else beta.rvs(dfi['alpha'].values, dfi['beta'].values, size=ns[-1])),dfi['p_use']*(1+dfi['b1']),(1-dfi['p_use'])*(1+dfi['b2']))))
    l.append([np.exp(w_opt[ni-1]-w_sigma[ni-1]) for ni in ns]+[1*(np.argmax(w_opt>np.log(wi)) <= np.argmax(w_sigma>np.log(wi))) for wi in ws])

pd.concat([pd.DataFrame(l).loc[:,:2].median(axis=0).reset_index(drop=True),pd.DataFrame(l).loc[:,3:].mean(axis=0).reset_index(drop=True)],axis=0)
