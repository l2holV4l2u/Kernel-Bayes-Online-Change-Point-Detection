import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.special import logsumexp
import pandas as pd

"""
def generate_data(mean0, var0, T, cp_prob) :
    data = []
    cps = []
    meanx = mean0
    cnt = 0
    for t in range(0, T) :
        if np.random.random() < cp_prob and cnt>=50 :
            if np.random.normal(0,1)>=0 :
                meanx = np.random.normal(meanx+1000, var0)
            else :
                meanx = np.random.normal(meanx-1000, var0)    
            cps.append(t)
            cnt=0
        if np.random.normal(0,1)>=0 :
            data.append(np.random.normal(meanx+50,var0))
        else :
            data.append(np.random.normal(meanx-50,var0))
        cnt+=1
    return data, cps, [0]+data
"""

def generate_data(T):
    df = pd.read_excel('Data3.xls')
    data = []
    cps = []
    for i in range(1100,T+1100):
        data.append(df.iloc[i,0]/500)
    return data, cps, [0]+data 

T      = 5000     # Number of observations.
hazard = 1/200  # Constant prior on changepoint probability.
mean0  = 0      # The prior mean on the mean parameter.
var0   = 2      # The prior variance for mean parameter.
varx   = 1      # The known variance of the data.

def bocd_kernel(hazard):
    log_R_kernel       = -np.inf * np.ones((T+1, T+1))
    log_R_kernel[0, 0] = 0       # log 0 == 1
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)
    cpk = []
    bef = 0
    for t in range(1, T+1):
        # Evaluate predictive probabilities.
        log_pis = kernel(t)
        # Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH
        # Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)
        # Calculate evidences
        log_message = np.append(log_cp_prob, log_growth_probs)
        # Determine run length distribution.
        log_R_kernel[t, :t+1]  = log_message - logsumexp(log_message)
        # Detector
        for i in range(1,t+1):
            if np.exp(log_R_kernel[t,i])>=0.000001 and np.exp(log_R_kernel[t-1,i-1]) < 0.000001 :
                log_R_kernel[t,i]=-np.inf
        for i in range(t,-1,-1):
            if np.exp(log_R_kernel[t,i])>=0.005:
                if bef-i>=10 :
                    if len(cpk)==0:
                        cpk.append(t)
                    elif t-cpk[len(cpk)-1]>5:
                        cpk.append(t)
                bef = i
                break
    return np.exp(log_R_kernel) , cpk

sqpi = np.sqrt(2*np.pi)

def normpdf(x):
    return (np.e**(-(x*x)/2))/sqpi
 
def kernel(t):
    prob = normpdf(nd[t-1]-nd[t])     
    pis = [np.log(prob)]
    for i in range(2,t+1):
        prob += normpdf(nd[t-i]-nd[t])
        pis.append(np.log(prob)-np.log(i))
    return np.array(pis)

def plot_posterior(T, data, cps, cpk, R_kernel):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))
    ax1, ax2 = axes
    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    ax1.title.set_text('Data')
    # Plot prediction for kernel version
    ax2.imshow(np.rot90(R_kernel), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.set_ylim([T,T-1000])
    ax2.margins(0)
    ax2.title.set_text('Kernel')
    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
    use = np.ones(len(cps))
    tp = fp = tn = fn = 0
    delay = []
    for i in range(len(cpk)):
        ax2.axvline(cpk[i], c='blue', ls='dotted')    
        flag=0
        for j in range(len(cps)):
            if cps[j]-cpk[i]<=20 and use[j]:
                flag=1
                use[j]=0
                delay.append(cpk[i]-cps[j])
                break
        if flag:
            tp+=1
        else :
            fp+=1
    for cnt in use:
        fn+=cnt
    tn=T-(tp+fp+fn)
    print("tp fp tn fn =",tp,fp,tn,fn)
    print("Accuracy :",((tp+tn)*100)/T)
    print(delay)
    
    plt.show()

#data, cps, nd = generate_data(mean0, var0, T, hazard)
data, cps, nd = generate_data(T)
R_kernel , cpk = bocd_kernel(hazard)
plot_posterior(T, data, cps, cpk, R_kernel)