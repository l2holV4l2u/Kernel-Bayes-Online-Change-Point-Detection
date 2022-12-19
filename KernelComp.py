import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.special import logsumexp
from   scipy.stats import norm

def generate_data(mean0, var0, T, cp_prob) :
    data = []
    cps = []
    meanx = mean0
    cnt = 0
    for t in range(0, T) :
        if np.random.random() < cp_prob and cnt>=50 :
            if np.random.normal(0,1)>=0 :
                meanx = np.random.normal(meanx+20, var0)
            else :
                meanx = np.random.normal(meanx-20, var0)    
            cps.append(t)
            cnt=0
        if np.random.normal(0,1)>=0 :
            data.append(np.random.normal(meanx+2,var0))
        else :
            data.append(np.random.normal(meanx-2,var0))
        #data.append(np.random.normal(meanx,var0))       
        #data.append(np.random.beta(5,2)+meanx)
        cnt+=1
    return data, cps, [0]+data

T      = int(input())     # Number of observations.
hazard = 5/100   # Constant prior on changepoint probability.
mean0  = 0      # The prior mean on the mean parameter.
var0   = 2      # The prior variance for mean parameter.
varx   = 1      # The known variance of the data.

def bocd(data, model, hazard):
    log_R       = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0              # log 0 == 1
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)
    cpg = []
    bef = 0
    for t in range(1, T+1):
        # Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, data[t-1])
        # Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH
        # Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)
        # Calculate evidence
        log_message = np.append(log_cp_prob, log_growth_probs)
        # Determine run length distribution.
        log_R[t, :t+1]  = log_message - logsumexp(log_message)
        # Update sufficient statistics.
        model.update_params(data[t-1])
        #detector
        for i in range(t,-1,-1):
            if np.exp(log_R[t,i])>=0.001:
                if bef-i>=5:
                    cpg.append(t)
                bef = i
                break
    return np.exp(log_R) , cpg

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
    pis = [prob]
    for i in range(2,t+1):
        prob += normpdf(nd[t-i]-nd[t])
        pis.append(prob/i)
    return np.array(np.log(pis)) 

class GaussianUnknownMean:
    def __init__(self, mean0, var0, varx):
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    def log_pred_prob(self, t, x):
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    def update_params(self, x):
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)
    @property
    def var_params(self):
        return 1./self.prec_params + self.varx

def plot_posterior(T, data, cps, cpk, cpg, R_kernel, R):
    fig, axes = plt.subplots(3, 1, figsize=(20,10))
    ax1, ax2, ax3 = axes
    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    ax1.title.set_text('Data')
    # Plot prediction for kernel version
    ax2.imshow(np.rot90(R_kernel), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)
    ax2.title.set_text('Kernel')
    # Plot prediction for bocd
    ax3.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax3.set_xlim([0, T])
    ax3.margins(0)
    ax3.title.set_text('BOCD')

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
    #    ax2.axvline(cp, c='red', ls='dotted')
        ax3.axvline(cp, c='red', ls='dotted')
    
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
    
    #for cp in cpg:
    #    ax3.axvline(cp, c='green', ls='dotted')
    
    plt.show()

data, cps, nd = generate_data(mean0, var0, T, hazard)
R_kernel , cpk = bocd_kernel(hazard)
model    = GaussianUnknownMean(mean0, var0, varx)
R , cpg = bocd(data, model, hazard)
plot_posterior(T, data, cps, cpk, cpg, R_kernel,R)