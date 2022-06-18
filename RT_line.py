import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import notebook
import time
from numba import jit,njit
import plotly.express as px
import pandas as pd
import plotly.subplots as sp
from plotly.offline import plot


@njit(parallel=True)
def Raylegh_Taylor(h=0.01, hh=0.01, t=0.008, k_inR=16, k_outR=1, g_first=1, g_second=1):
    """
    Solution of the set of equations modeling Rayleigh-Taylor instability in a porous medium.
    
    Parameters
    ----------
    h:  float
        step on the x-axis.
    hh: float
        step on the y-axis.
    t:  float
        step on the time.
    k_inR:  float
        coefficient inside the R.
    k_outR:  float
        coefficient in front of the R.
    g_first:  float
        coefficient in front of the R for p < (pp + pm) / 2.
    g_second:  float
        coefficient in front of the R for p > (pp + pm) / 2.
        
    Returns
    -------
    p:  array
        values of density in the grid.
    Pr: array
        values of pressure in the grid

    """
    k1 = 0.02
    k2 = 0.01
    k3 = 0.017
    k4 = k_outR 
    k5 = 1
    k6 = k_inR 

    pp = 1.2
    pm = 1           

    t1 = 1+int(1/t) 
    hh1 = 1+int(1/hh) 
    h1 = 1+int(1/h) 
    p = np.zeros((h1,hh1,t1))
    Pr = np.zeros((h1,hh1,t1))

    u = np.zeros((h1,hh1,t1))
    v = np.zeros((h1,hh1,t1))
    R = np.zeros((h1,hh1,t1))

    AA = np.zeros((h1,hh1,t1))
    BB = np.zeros((h1,hh1,t1))
    CC = np.zeros((h1,hh1,t1))
    F = np.zeros((h1,hh1,t1))
    alpha = np.zeros((h1,hh1,t1))
    beta = np.zeros((h1,hh1,t1))
    
    AA_ = np.zeros((h1,hh1,t1))
    BB_ = np.zeros((h1,hh1,t1))
    CC_ = np.zeros((h1,hh1,t1))
    F_ = np.zeros((h1,hh1,t1))
    alpha_ = np.zeros((h1,hh1,t1))
    beta_ = np.zeros((h1,hh1,t1))

    for i in range(0,h1): 
        for k in range(0,hh1):
            if(k>=(hh1-1)/2):
                p[i,k,0] = pp
                Pr[i,k,0] = k3/k1 * pp * ((hh1-1) * hh - k*hh)
            else:
                p[i,k,0] = pm
                Pr[i,k,0] = k3/k1 * (pp * (hh1-1) * hh/2 + pm * ((hh1-1) * hh/2 - k*hh))
            

    #for j in notebook.tqdm(range(1,t1)):
    for j in range(1,t1):

            for i in range(1,h1-1): 

                for k in range(1,hh1-1):
                
                    if(j > 1):
                        if(k == 1):
                            Pr[i,0,j-1] = Pr[i,0,0]

                        Pr[i,k,j-1] = alpha_[i,k,j-1] * Pr[i,k-1,j-1] + beta_[i,k,j-1]
                        
                        if(i == 1):
                            Pr[0,k,j-1] = Pr[1,k,j-1]
                            if(k == 1):
                                Pr[0,0,j-1] = Pr[1,0,j-1]

                        if(i == h1-2):
                            Pr[h1-1,k,j-1] = Pr[h1-2,k,j-1]
                            if(k == 1):
                                Pr[h1-1,0,j-1] = Pr[h1-2,0,j-1]

                        if(k == hh1-2):
                            Pr[i,hh1-1,j-1] = Pr[i,hh1-2,j-1] - k3/k1 * p[i,hh1-2,j-1]*hh
                            if(i == h1-2):
                                Pr[h1-1,hh1-1,j-1] = Pr[h1-2,hh1-1,j-1]
                            if(i == 1):
                                Pr[0,hh1-1,j-1] = Pr[1,hh1-1,j-1]
            

                    u[i,k,j-1] = (-k1*(Pr[i+1,k,j-1] - Pr[i,k,j-1])/h)
              
                    v[i,k,j-1] = -k1*(Pr[i,k,j-1] - Pr[i,k-1,j-1])/hh - k3*p[i,k,j-1]  
                    

                    R[i,k,j-1] =k4*((-pm + 2*p[i,k,j-1] - pp) * (1-k6*(p[i,k,j-1] - pm) * (p[i,k,j-1] - pp) / ((pp - pm)**2)))

                    
                    if(k == 1):
                        R[i,0,j-1] = k4*((-pm + 2*p[i,0,j-1] - pp)*(1-k6*(p[i,0,j-1]-pm)*(p[i,0,j-1]-pp)/((pp-pm)**2)))
                        p[i,0,j] = pm
                        if(i == 1):
                            p[0,0,j] = pm
                            p[0,hh1-1,j] = pp
                            p[h1-1,0,j] = pm
                            p[h1-1,hh1-1,j] = pp

                    if(k == hh1-2):    
                        R[i,hh1-1,j-1] = k4*((-pm + 2*p[i,hh1-1,j-1] - pp)*(1-k6*(p[i,hh1-1,j-1]-pm)*(p[i,hh1-1,j-1]-pp)/((pp-pm)**2)))
                        p[i,hh1-1,j] = pp
          

                    if ((pm <= p[i,k-1,j-1]) & (p[i,k-1,j-1] <= (pp+pm)/2)):
                        Gk = g_first    
                    else:
                        Gk = g_second
                        
                    if ((j/(t1-1)) <= 0.4):
                        gg = 0
                    else:
                        gg = 1

                    if(k == 1):
                        alpha[i,0,j] = 0
                        beta[i,0,j] = pm

                    AA[i,k-1,j] = (
                                     -Gk*k5*t/hh*(R[i,k-1,j-1] + np.abs(R[i,k-1,j-1]))/2
                                     -k2*t/hh*(v[i,k-1,j-1] + np.abs(v[i,k-1,j-1]))/2
                                    )
                    BB[i,k-1,j] = (
                                     Gk*k5*t/hh*(R[i,k-1,j-1] - np.abs(R[i,k-1,j-1]))/2
                                     +k2*t/hh*(v[i,k-1,j-1] - np.abs(v[i,k-1,j-1]))/2
                                    )


                    CC[i,k-1,j] = - (
                                      Gk*k5*t/hh*(R[i,k-1,j-1] + np.abs(R[i,k-1,j-1]))/2 - Gk*k5*t/hh*(R[i,k-1,j-1] - np.abs(R[i,k-1,j-1]))/2
                                      +k2*t/hh*(v[i,k,j-1] - v[i,k-1,j-1])
                                      +1
                                      #+gg*t #add
                                      )
                    F[i,k-1,j] =    -(
                                    p[i,k-1,j-1]
                                    #+gg*pm*t #add
                                    )

                    if(k == hh1-2):
                        AA[i,hh1-2,j] = (
                                           -Gk*k5*t/hh*(R[i,hh1-2,j-1] + np.abs(R[i,hh1-2,j-1]))/2
                                           -k2*t/hh*(v[i,hh1-2,j-1] + np.abs(v[i,hh1-2,j-1]))/2
                                          )
                        BB[i,hh1-2,j] = (
                                           Gk*k5*t/hh*(R[i,hh1-2,j-1] - np.abs(R[i,hh1-2,j-1]))/2
                                           +k2*t/hh*(v[i,hh1-2,j-1] - np.abs(v[i,hh1-2,j-1]))/2
                                          )
                        CC[i,hh1-2,j] = -(
                                           Gk*k5*t/hh*(R[i,hh1-2,j-1] + np.abs(R[i,hh1-2,j-1]))/2 - Gk*k5*t/hh*(R[i,hh1-2,j-1] - np.abs(R[i,hh1-2,j-1]))/2
                                           +k2*t/hh*(v[i,hh1-1,j-1] - v[i,hh1-2,j-1])
                                           +1
                                           #+gg*t #add
                                           )
                        F[i,hh1-2,j] =  -(
                                        p[i,hh1-2,j-1]
                                        #+gg*pm*t #add
                                        )


                    alpha[i,k,j] = BB[i,k-1,j] / (CC[i,k-1,j] - alpha[i,k-1,j] * AA[i,k-1,j])
                    beta[i,k,j] = (AA[i,k-1,j] * beta[i,k-1,j] + F[i,k-1,j]) / (CC[i,k-1,j] - alpha[i,k-1,j] * AA[i,k-1,j])

                    if(k == hh1-2):

                        alpha[i,hh1-1,j] = BB[i,hh1-2,j] / (CC[i,hh1-2,j] - alpha[i,hh1-2,j] * AA[i,hh1-2,j])
                        beta[i,hh1-1,j] = (AA[i,hh1-2,j] * beta[i,hh1-2,j] + F[i,hh1-2,j]) / (CC[i,hh1-2,j] - alpha[i,hh1-2,j] * AA[i,hh1-2,j])
                             
                
                for k in range(hh1-2,0,-1): 

                    p[i,k,j] = alpha[i,k+1,j] * p[i,k+1,j] + beta[i,k+1,j]
                    if(i == 1):
                        p[0,k,j] = p[1,k,j]
                    if(i == h1-2):
                        p[h1-1,k,j] = p[h1-2,k,j]

                for k in range(hh1-2,0,-1): 

                    if(k == 1):
                        alpha_[i,hh1-1,j] = 1
                        beta_[i,hh1-1,j] = - k3/k1 * p[i,hh1-1,j] * hh

                    AA_[i,k,j] =  (
                                    1/hh**2
                                    )
                    BB_[i,k,j] =  (
                                    1/hh**2
                                    )
                    CC_[i,k,j] =  -( 
                                    - 2/hh**2
                                    )
                    F_[i,k,j] =   k3/k1 * (
                                    (p[i,k,j]-p[i,k-1,j])/hh
                                    )

                    if(k == 1):
                        AA_[i,0,j] = (
                                        1/hh**2
                                        )
                        BB_[i,0,j] = (
                                        1/hh**2
                                        )
                        CC_[i,0,j] = -( 
                                        - 2/hh**2
                                        )
                        F_[i,0,j] =   k3/k1 * (
                                        (p[i,1,j]-p[i,0,j])/hh
                                        )

                    alpha_[i,k,j] = AA_[i,k,j] / (CC_[i,k,j] - alpha_[i,k+1,j] * BB_[i,k,j])
                    beta_[i,k,j] = (BB_[i,k,j] * beta_[i,k+1,j] + F_[i,k,j]) / (CC_[i,k,j] - alpha_[i,k+1,j] * BB_[i,k,j])
                    if(k == 1):
                        alpha_[i,0,j] = AA_[i,0,j] / (CC_[i,0,j] - alpha_[i,1,j] * BB_[i,0,j])
                        beta_[i,0,j] = (BB_[i,0,j] * beta_[i,1,j] + F_[i,0,j]) / (CC_[i,0,j] - alpha_[i,1,j] * BB_[i,0,j])
                    
                    
    return p, Pr

# +
start = time.time()
Raylegh_Taylor(k_inR = 16, g_first = 1, g_second = 1)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

start = time.time()
Raylegh_Taylor(k_inR = 16, g_first = 1, g_second = 1)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


# -

def Ani_Density(p, Pr):
    N_y = len(p[5,:,0])
    N_t = len(p[5,0,:])
    t_for_graph = np.repeat(np.array(range(0,N_t)),N_y)
    y_for_graph = np.tile(np.array(range(0,N_y)),N_t)/100
    p_for_graph = np.ravel(p[5,:,:], order = 'F')
    df = pd.DataFrame(data = {'p' : p_for_graph, 'y' : y_for_graph, 't' : t_for_graph})
    
    fig = px.area(df, y='y', x = 'p', animation_frame = 't',
            range_x=[0.998,1.202], range_y=[-0.002,1.002], width=450, height=680,markers=True,
        color_discrete_sequence = ["darkslateblue", "coral"], title="Average Density")
    
    fig.update_layout(plot_bgcolor = 'snow')
    
    fig.show()


p, Pr = Raylegh_Taylor(k_inR=16, g_first=1, g_second=1)
Ani_Density(p,Pr)


