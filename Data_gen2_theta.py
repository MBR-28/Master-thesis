import autograd.numpy as np
import random as ran
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import autograd
#import autograd.numpy as np

#importing multiprocessing to speed up data generation
import concurrent.futures as cf

#timer
from time import time


#Defining constant Global parameters
m = 1/2 #kg
g = 9.81 # m/s**2
L = 1 #m
'''
#Lagrangian
def Lagrangian(koord):
    q,dq = np.split(koord,2)
    lag= 0.5*m*dq*dq - m*g*np.sin(q/L)
    return lag
'''

#Define Hamiltonian formula (q,p)

def Hamiltonian(koord):
    q,p = np.split(koord,2)
    H = (1/(2*m))*p*p + m*g*np.sin(q/L)
    return H



#Define S(q,p) 

def dynamics(t,koord):
    #print(koord)
    #print(np.shape(koord))
    if np.shape(koord)[0]==1:
        #print("H")
        koord = np.reshape(koord,(2,))
        #print(koord)
    
    q,dq = np.split(koord,2)

    
    '''
    #tt = autograd.grad(Lagrangian)(q,dq)
    tt = autograd.grad(Lagrangian)(koord)
    t2,t1 = np.split(tt,2)
    tt2 = np.gradient(t2)
    #print(tt2)
    #print("AAAAAA")
    tau = np.array([tt2 - t1])
    #print(tau)
    #print(np.shape(tau))
    
    '''
    #tau based on spring dampner
    kp = 5
    kd =-7
    th0 = np.pi/2

    tau = kp*(th0-(q/L)) + kd*(dq/L)


    p = m*dq

    koord2 = np.array([q,p])

    dH = autograd.grad(Hamiltonian)(koord2)
    dqdt, dpdt = np.split(dH,2)

    
    #S=np.concatenate([dqdt,-dpdt+tau],axis=1)
    S=np.concatenate([dqdt,(-dpdt+tau)/m],axis=1)
    return S


def no_tau_dynamics(t,koord):
    #print(koord)
    #print(np.shape(koord))
    if np.shape(koord)[0]==1:
        #print("H")
        koord = np.reshape(koord,(2,))
        #print(koord)
    
    q,dq = np.split(koord,2)

    p = m*dq

    koord2 = np.array([q,p])

    dH = autograd.grad(Hamiltonian)(koord2)
    dqdt, dpdt = np.split(dH,2)

    #S=np.concatenate([dqdt,-dpdt],axis=1)
    S = np.concatenate([dqdt,-dpdt/m],axis=1)
    return S


def trajectory(t_span=[0,3],t_scale=10,noise=0.1,tolerance = 1e-10):
    t0,t1 = t_span[0],t_span[1]
    eval_time = np.linspace(t0,t1,int(t_scale*(t1-t0)))
    #Genetrate an initial angle within reasonable limits
    #Angle 0 to pi radians
    n = 100
    theta0 = (ran.randint(0,n)/n)*np.pi

    q0 = L*theta0
    #concatenate to q

    #Generer tilfeldig rot hastighet
    dtheta0 = (ran.randint(-10*n,10*n)/n)
    #bruk dtheta til å finne dq1
    
    dq0 = L*dtheta0
    

    qdq0 = np.array([q0,dq0])
    #print(qdq0)

    #bruk S til å beregne dqdt og dpdt 
    #using no tau as gradient doesn't work for single values
    arm_ivp = solve_ivp(fun=dynamics,t_span=t_span,y0=qdq0,t_eval=eval_time,rtol=tolerance)
    qdq1 = np.array([arm_ivp['y'][0], arm_ivp['y'][1]])

    q1,dq1 = np.split(qdq1,2)
    
    q,dq = [],[]
    #Removal of excess list nesting
    for l in range(len(q1[0])):
        tempq = np.array([q1[0][l]])
        q.append(tempq)
        tempdq = np.array([dq1[0][l]])
        dq.append(tempdq)
        
    dqdt,dpdt = [],[]
    #Calculation of gradients due to solve_ivp not outputing gradients.
    for i in range(len(q)):
        x = np.array([q[i][0],dq[i][0]])
        dyn = dynamics(None,x)
        #print(dyn)
        dqdt.append(np.array([dyn[0][0]]))
        dpdt.append(np.array([dyn[0][1]]))
    
    q = np.array(q)
    dq = np.array(dq)
    dqdt = np.array(dqdt)
    dpdt = np.array(dpdt)

   

    #sammenligne dq1 og dqdt for å se om de er den samme?
    #korriger?
    #if abs(dq-dqdt).any() > 1e-5:
    #    dqdt = dq
   

    #add noise to input values and export together with corresponding output values

    q += ran.random()*noise
    dq += ran.random()*noise

    return q,dq, dqdt, dpdt


def no_tau_trajectory(t_span=[0,3],t_scale=10,noise=0.1,tolerance = 1e-10):
    t0,t1 = t_span[0],t_span[1]
    eval_time = np.linspace(t0,t1,int(t_scale*(t1-t0)))

    #print("Rand")
    #Genetrate an initial angle within reasonable limits
    #Angle 0 to pi radians
    n = 100
    theta0 = (ran.randint(0,n)/n)*np.pi

    q0 = L*theta0
    #concatenate to q

    #Generer tilfeldig rot hastighet
    dtheta0 = (ran.randint(-10*n,10*n)/n)
    #bruk dtheta til å finne dq1
    
    dq0 = L*dtheta0
    

    #print("combine")
    qdq0 = np.array([q0,dq0])
    #print(qdq0)

    #bruk S til å beregne dqdt og dpdt 
    #print("IVP")
    arm_ivp = solve_ivp(fun=no_tau_dynamics,t_span=t_span,y0=qdq0,t_eval=eval_time,rtol=tolerance)
    #print("split")
    qdq1 = np.array([arm_ivp['y'][0], arm_ivp['y'][1]])

    q1,dq1 = np.split(qdq1,2)
    
    
    #print("Arrange")
    q,dq = [],[]
    
    for l in range(len(q1[0])):
        tempq = np.array([q1[0][l]])
        q.append(tempq)
        tempdq = np.array([dq1[0][l]])
        dq.append(tempdq)
        
    dqdt,dpdt = [],[]
    #print("dynamic")

    for i in range(len(q)):
        x = np.array([q[i][0],dq[i][0]])
        dyn = no_tau_dynamics(None,x)
        dqdt.append(np.array([dyn[0][0]]))
        dpdt.append(np.array([dyn[0][1]]))
    
    q = np.array(q)
    dq = np.array(dq)
    dqdt = np.array(dqdt)
    dpdt = np.array(dpdt)

    

    #sammenligne dq1 og dqdt for å se om de er den samme?
    #korriger?
    #if abs(dq-dqdt).any() > 1e-5:
    #    dqdt = dq
   

    #add noise to input values and export together with corresponding output values
    #print("Noise")
    q += ran.random()*noise
    dq += ran.random()*noise

    return q,dq, dqdt, dpdt




'''
Worker for multiprcessing
'''
con = 0

def trajectory_worker(t_span,t_scale,noise, tolerance):
    q, dq, dqdt, dpdt = trajectory(t_span=t_span,t_scale=t_scale,noise=noise,tolerance = tolerance)
    return q, dq, dqdt, dpdt



def no_tau_trajectory_worker(t_span,t_scale,noise, tolerance):
    
    #print("Non_tau")
    q, dq, dqdt, dpdt = no_tau_trajectory(t_span=t_span,t_scale=t_scale,noise=noise,tolerance = tolerance)
    #print("Non_tau_done")
    return q, dq, dqdt, dpdt 




"""
Split the data to make a training and a test set 
format as needed for training of the neural network
"""
def dataset_theta(multiprocessing=False, seed=None,samples=60,split_point = 30,t_span=[0,3],t_scale=10,noise=0.1,tolerance = 1e-10,tau=False):
    print("Creating dataset")
    data = {'meta':locals()}
    ran.seed(seed)


    qp, dqp = [], []

    #generate "sample" number of trajectories with derivatives
    print_every = int(samples//200)
    if print_every == 0:
        print_every = 1


    if True==multiprocessing:
        print("Starting multiprocessing")
        t0 = time()
        MultiProc = []

        with cf.ProcessPoolExecutor(max_workers=6) as executor: 

            if tau==True:
                for j in range(samples):
                    if j%10000==0:
                        print(f"{j} processes started")
                        
                    proc = executor.submit(trajectory_worker,t_span,t_scale,noise, tolerance)
                    MultiProc.append(proc)
                print(f"{j} processes started")
            else:
                #print("CCC")
                for j in range(samples):
                    if j%10000==0:
                        print(f"{j} processes started")
                    
                    proc = executor.submit(no_tau_trajectory_worker,t_span,t_scale,noise, tolerance)
                    MultiProc.append(proc)
                print(f"{j} processes started")
            
            print("Finishing processes")
            counter = 0
            for t in cf.as_completed(MultiProc):
                #print("Finishing")
                q,dq,dqdt,dpdt = t.result()

                qp.append(np.array([q[0],dq[0]]).T)
                dqp.append(np.array([dqdt[0],dpdt[0]]).T)

                counter+=1
                if counter % print_every == 0:
                    t1 = time()
                    frac = (counter/samples)
                    percentage = frac*100
                    time_elapsed = t1-t0 
                    time_from_start_to_end = time_elapsed/frac
                    time_til_end = time_from_start_to_end-time_elapsed
                    print("")
                    print(f"Processes finished: {counter}, Percentage of total: {percentage:.1f}")
                    print(f"Time elapsed: {time_elapsed/60:.2f} min. Estimated time until finished: {time_til_end/60:.2f} min")
        
        print("Multiprocessing finished")


    else:
        print("Starting sampling")
        t0 = time()

        if True==tau:
                for j in range(samples):

                    q, dq, dqdt, dpdt = trajectory(t_span=t_span,t_scale=t_scale,noise=noise,tolerance = tolerance)
                    #print(np.shape(q))
                    #print(np.shape(dq))
                    #print(np.shape(dqdt))
                    #print(np.shape(dpdt))
                    qp.append(np.array([q[0],dq[0]]).T)
                    dqp.append(np.array([dqdt[0],dpdt[0]]).T)


                    if (j % print_every == 0) and (j!=0):
                        t1 = time()
                        frac = (j/samples)
                        percentage = frac*100
                        time_elapsed = t1-t0 
                        time_from_start_to_end = time_elapsed/frac
                        time_til_end = time_from_start_to_end-time_elapsed
                        print("")
                        print(f"Processes finished: {j}, Percentage of total: {percentage:.1f}")
                        print(f"Time elapsed: {time_elapsed/60:.2f} min. Estimated time until finished: {time_til_end/60:.2f} min")
        
        else:
        


            for j in range(samples):

                q, dq, dqdt, dpdt = no_tau_trajectory(t_span=t_span,t_scale=t_scale,noise=noise,tolerance = tolerance)
                #print(np.shape(q))
                #print(np.shape(dq))
                #print(np.shape(dqdt))
                #print(np.shape(dpdt))
                qp.append(np.array([q[0],dq[0]]).T)
                dqp.append(np.array([dqdt[0],dpdt[0]]).T)


                if (j % print_every == 0) and (j!=0):
                    t1 = time()
                    frac = (j/samples)
                    percentage = frac*100
                    time_elapsed = t1-t0 
                    time_from_start_to_end = time_elapsed/frac
                    time_til_end = time_from_start_to_end-time_elapsed
                    print("")
                    print(f"Processes finished: {j}, Percentage of total: {percentage:.1f}")
                    print(f"Time elapsed: {time_elapsed/60:.2f} min. Estimated time until finished: {time_til_end/60:.2f} min")


    #print(qp)


    #randomizing
    print("Randomizing")
    xy,dxy = [],[]

    for l in range(len(qp)):
        length = len(qp)
        i = np.random.randint(0,length)
            
        x = qp.pop(i)
        dx = dqp.pop(i)

        xy.append(x)
        dxy.append(dx)


    data['qp'] = np.concatenate(xy)
    data['dqp'] = np.concatenate(dxy)
    
    

    
    
    split = {}

    print("Splitting dataset")
    for k in ['qp','dqp']:
        #Train set everything before split point, test set after
        split[k], split['test_'+k] = data[k][split_point:],data[k][:split_point]
    print("Finished data set")
    return split
    

