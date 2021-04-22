
# Libraries
import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
plt.close("all")                                                               
import pandas as pd




# OUTLINE OF THE SCRIPT:
# 1- simulations to help construct the fixed effect (FE) and random effects (RE) models
# 2- FE model
# 3- RE model
# 4- Functions that estimate the paramaters of the models
# 5- Real data extracted from Bornestein et al. (2015) p. 152 Table 19.1
# 6- Run the two models for the data
# 7- Funnel plot of the data
# 8- Simulated effect size and standard errors that show funnel plot asymmetry
# 9- Funnel plot of these simulations
# 10- Run the two models on the simulations, to be compared with the trim and fill adjusted estimates later
# 11- trim and fill adjustment after Rothstein, Sutton & Borenstein (2005) chapter 8, functions that estimate the parameters of the model
# 12- Very messy but I constructed the functions in #11 based on this part so it was necessary
# 13- pandas dataframe of most relevant steps
# 14- Funnel plot filled with the imputed data




#1 simulations
k = 40 # number of studies
true_ES = 0 # fixed true but unknown effect size or true but unknown expected value of random effect sizes
E_V = 1 # expected value of true but unknown within-study variance 
ES = np.random.normal(true_ES, E_V, k) # 40 random effect sizes from a standard normal
V = np.random.chisquare(1, 40) # 40 random within study variances from a Chi square with 1 df




#2 fixed effect model - after Cooper, Hedges & Valentine (2019) Chapter 11-12, see below for implementation as functions
mini_w = 1/V # variance minimising weights
F_ES_est = np.sum(ES*mini_w)/np.sum(mini_w) # weighted fixed ES estimate
con_V = 1/np.sum(1/V) # conditional variance of the fixed ES estimate
SE_F = np.sqrt(con_V) # standard error of the fixed effect size estimate
t39 = 2.022691 # two tailed t-value at k-1 degrees of freedom
CI = t39*SE_F # Adding this product to and substracting it from the ES estimate yields the 95% confidence interval
homog_test = np.sum(mini_w*np.square(ES))-(np.square(np.sum(mini_w*ES))/np.sum(mini_w)) # homogneiety test Q, fixed effects model is inappropriate if this test statistics exceeds the corresponding critical value.




#3 random effects model - after Cooper, Hedges & Valentine (2019) Chapter 11-12,see directly below for implementation as functions
V_comp_est = (homog_test-(k-1))/np.sum(mini_w)-(np.sum(np.square(mini_w))/np.sum(mini_w)) # variance component (between study variance) estimate, set to zero if negative.
if V_comp_est <= 0 : # set variance component to 0 if negative
    V_comp_est = 0
mini_w_R = 1/(V+V_comp_est) # variance minimising weights for random effects model
R_ES_est = np.sum(ES*mini_w_R)/np.sum(mini_w_R) # estimate of random effect size 




#4 Equations as functions
def DeltaHat(mini_w, ES):
    """  Estimate of fixed effect size or mean of random effects
    Inputs: variance minimising weights, effect sizes
    Output: Delta effect size summary statistic """
    Delta = np.sum(ES*mini_w)/np.sum(mini_w)
    return Delta
def Q(mini_w, ES):
    """ Homogeneity test. 
    Inputs: variance minimising weights, effect sizes
    Output: Q homogeneity test score """
    Q = np.sum(mini_w*np.square(ES))-(np.square(np.sum(mini_w*ES))/np.sum(mini_w))
    return Q
def TauSqHat(Q, mini_w, k):
    """ Estimates the between-study variance, aka variance component, set to 0 if negative
    Inputs: Q homogeniety test score, variance minimising weights, number of studies
    Output: Estimate of between-study variance Tau squared """
    if 0<(Q-(k-1))/(np.sum(mini_w)-(np.sum(np.square(mini_w))/np.sum(mini_w))):
        return (Q-(k-1))/(np.sum(mini_w)-(np.sum(np.square(mini_w))/np.sum(mini_w)))
    else:
        return 0




#5 Data from Bornestein et al. (2015) p. 152 Table 19.1
dk = 10
dESA = np.array([0.110, 0.224, 0.338, 0.451, 0.480]) # effect sizes from studies in group A
dESB = np.array([0.440, 0.492, 0.651, 0.710, 0.740]) # effect sizes from studies in group B
dES = np.concatenate([dESA,dESB]) # all effect sizes
dVA = np.array([0.01, 0.03, 0.02, 0.015, 0.01]) # within-study variances group A
dVB = np.array([0.015, 0.02, 0.015, 0.025, 0.012]) # within-study variances group B
dV = np.concatenate([dVA,dVB]) # all within-study variances
dSE = np.sqrt(dV) # study-level standard errors
dSE_recip = 1/dSE # reciprocals of standard errors. commonly used for funnel plots




#6 Testing the fixed and random effects models on the data
dmini_w = 1/dV
DeltaHat(dmini_w, dES) # 0.4581
Q(dmini_w, dES) # 26.4371
TauSqHat(Q(dmini_w, dES), dmini_w, dk) # 0.0299
dmini_w_R = 1/(dV + TauSqHat(Q(dmini_w, dES), dmini_w, dk))
DeltaHat(dmini_w_R, dES) # 0.4638




#7 Funnel plot 
plt.scatter(dES, dSE_recip, s = 49, edgecolor="black", linewidth=1, alpha=0.78)
plt.xlabel("Effect sizes")
plt.ylabel("1/Standard erors")
plt.title("Funnel plot")
plt.xlim(0)
plt.plot([0.2, 0.4], [0, 10], 'k-') # arbitrary line to draw the funnel shape
plt.plot([0.4, 0.6], [10, 0], 'k-') # arbitrary line to draw the funnel shape




#8 simulated effect sizes and standard errors for illustration purposes
np.random.seed(1) # fix the random samples for reproducibility
ds_cluster1 = np.random.normal(0.5,0.01, 15) 
rSEs_cluster1 = np.random.normal(8, 1, 15)
ds_cluster2 = np.random.normal(0.55,0.05, 10) 
rSEs_cluster2 = np.random.normal(5, 1, 10)
ds_cluster3 = np.random.normal(0.6,0.14, 5) 
rSEs_cluster3 = np.random.normal(2.5, 1, 5)
ds_cluster4 = np.random.normal(0.65,0.1, 20) 
rSEs_cluster4 = np.random.normal(1.5, 1, 20)
ds = np.concatenate([ds_cluster1, ds_cluster2, ds_cluster3, ds_cluster4]) # effect sizes
rSEs = np.concatenate([rSEs_cluster1, rSEs_cluster2, rSEs_cluster3, rSEs_cluster4]) # reciprocals of standard errors
ks= 50 # number of studies
SEs = 1/rSEs # standard errors
Vs = np.square(SEs) # within-study variances
ws = 1/Vs




#9 Funnel plot of made up data
plt.figure(figsize = (12,8))
plt.rcParams.update({'font.size': 19})
plt.scatter(ds, rSEs, s = 60, edgecolor="black", linewidth=1, alpha=0.78)
plt.xlabel("Effect sizes")
plt.ylabel("1/Standard errors")
plt.xlim(0, 1)
plt.ylim(0, 10)
plt.plot([0.35, 0.5], [0, 10], 'k-') # arbitrary line to draw the funnel shape
plt.plot([0.5, 0.65], [10, 0], 'k-') # arbitrary line to draw the funnel shape
plt.plot([DeltaHat(ws, ds), DeltaHat(ws, ds)], [0, 10], 'r-', linewidth=4) # Delta estimate as a red line
plt.savefig('FP1.png')




#10 applying the RE and FE models to the fake data, to be compared with the adjusted model later
uQ = Q(ws, ds) # heterogeneity test on the illustration data
uT = TauSqHat(uQ, ws, ks) # between-study variance estimate
ws_R = 1/(Vs + uT) # variance minimising weights for random effects model
R = DeltaHat(ws_R, ds) # random effects summary estimate = 0.5165 
F = DeltaHat(ws, ds) # fixed effect estimate = random effects estimate because between-study variance was set to 0 




#11 functions that estimate the parameters of the trim and fill method 
def f_srl(d, DeltaHat):
    '''evaluates the signed-rank list.
    inputs: original effect sizes, summary effect size estimate
    output: signed-rank list'''
    return ((np.absolute(np.sort(d)-DeltaHat)).argsort().argsort()+1) * np.sign(np.sort(d)-DeltaHat)

def f_Srank(srl):
    '''evaluates Srank, the sum of all positive values in the srl.'''
    return (sum(di for di in srl if di > 0))

def f_L0(Srank, k):
    '''estimates k0, the number of missing studies.
    inputs: Srank, number of studies
    output: L0 estimate of k0'''
    return int(round((4*Srank-k*(k+1))/(2*k-1)))

def f_R0(srl):
    '''another estimate of k0.
    inputs: srl.
    output: R0 estimate of k0'''
    return (len(np.split(srl, np.where(np.diff(srl) != 1)[0]+1)[-1]))-1

def f_td(d, k0Hat):
    '''returns the ESs minus rightmost k0Hat removed ESs.
    inputs: effect sizes, estimate of k0(L0 or R0)
    output: ...'''
    return np.delete(np.sort(d), np.sort(d).argsort()[-k0Hat:][::-1])

def f_tw(d, V, TauSqHat, k0Hat):
    ''' returns the weights corresponding to the remaining ESs after the trimming.
    inputs: ESs, within-study variances, estimate of between-studiy variance, estimate of k0
    output: ...'''
    return 1/((np.delete(V[d.argsort()[::1]], np.sort(d).argsort()[-k0Hat:][::-1])) + TauSqHat)

def f_fd(d, k0Hat, DeltaHat):
    '''returns the original ESs + the imputed symmetric ESs
    inputs: ESs, estimate of k0, summary effect size estimate
    output: ...'''
    return np.concatenate((d,(2*DeltaHat)-np.sort(d)[-k0Hat:][::-1]))

def f_fSE(d, SE, V, k0Hat):
    '''returns the original SEs + the imputed symmetric SEs
    inputs: ESs, SEs, within-study variances, estimate of k0
    output: ...'''
    return np.concatenate((SE, np.sqrt((V[d.argsort()[::1]])[-k0Hat:][::-1])))



#12 trim and fill adjustment after Rothstein, Sutton & Borenstein (2005) chapter 8, following steps at pages 137-138, 
# see below for implemetation as functions, functions were tested for the last iteration
dss = np.sort(ds) #1 ESs sorted
Vss = Vs[ds.argsort()[::1]] #1 allocate corresponding variances 
#2 DeltaHat computed above
cen_ds = dss-F #3 effect sizes centered around the summary effect size estimate 
s1_cen_ds = np.where(cen_ds>0, 1, cen_ds*(1/-cen_ds)) #4 positive and negative deviations set to signed ones, to be multiplied with the ranked array.np.sign discovered later
abs_cen_ds = np.absolute(cen_ds) #4 absolute ES deviations
ran_abs_cen_ds = abs_cen_ds.argsort().argsort() + 1 #4 rank the absolute deviations starting with one
srl = ran_abs_cen_ds * s1_cen_ds #4 signed-rank list of centered ESs
Srank = sum(di for di in srl if di > 0) #5 initial value of Srank, the sum of all positive values in the srl
L0 = int(np.ceil((4*Srank-ks*(ks+1))/(2*ks-1))) #6 estimate of k0, i.e. the number of missing studies = 12.54, round up or down
gamma_star = len(np.split(srl, np.where(np.diff(srl) != 1)[0]+1)[-1]) #6 length of the rightmost run of consecutive ranks associated with positive values centered ES
R0 = gamma_star-1 #6 2nd estimate of k0, i.e. the number of missing studies = 17
tds = np.delete(dss, dss.argsort()[-L0:][::-1]) #7 sorted effect size minus the trimmed rightmost 13=L0
tws = 1/(np.delete(Vss, dss.argsort()[-L0:][::-1])) #7 corresponding weights
tF = DeltaHat(tws, tds) #7 fixed effect estimate with the 37 remaining ESs = 0.4966 < F
tcen_ds = np.sort(dss-tF) #8 original data centered around new DeltaHat
s1_tcen_ds = np.where(tcen_ds>0, 1, tcen_ds*(1/-tcen_ds)) #8 same procedure as before
abs_tcen_ds = np.absolute(tcen_ds) #8
ran_abs_tcen_ds = abs_tcen_ds.argsort().argsort() # 8
tsrl = ran_abs_tcen_ds * s1_tcen_ds #8
tSrank = sum(di for di in tsrl if di > 0) #9 Srank for the trimmed data
tL0 = int((4*tSrank-ks*(ks+1))/(2*ks-1)) #9 new estimate = 13.39, removed np.ceil function to round down to 13, round() discovered later
tgamma_star = len(np.split(tsrl, np.where(np.diff(tsrl) != 1)[0]+1)[-1]) #9
tR0 = tgamma_star-1 #9 new R0 estimate = 18
t2ds = f_td(ds, tL0) #10 trim the n=tL0 rightmost values of the original array of ESs
t2ws = f_tw(ds, Vs, 0, tL0) #10
t2F = DeltaHat(t2ws, t2ds) #10
t2srl = f_srl(ds, t2F) #10
t2Srank = f_Srank(t2srl) #10
t2L0 = f_L0(t2Srank, ks) #10 = 15 != tL0 (previous estimate). should be continued but no more time.
t2R0 = f_R0(t2srl) # 10 = tR0 = 18
#11 iterate the centering, ranking and estimating until estimates of Srank or L0 stabilise, i.e. Srank[k] = Srank[k-1]
#12 after Duval and Tweedie, 2000, to the original data add n = converged L0/R0 symmetric imputed ESs and associated SEs, 
#here for illustration purposes using t2R0 as estimate for k0
ids = (2*t2F)-(dss[-t2R0:][::-1]) #12 imputed effect sizes
fds = np.concatenate((ds, ids)) #12 the "fill" part in "trim and fill" 
fds = f_fd(ds, t2R0, t2F) # 12 impute and fill function for ESs
iSEs = np.sqrt(Vss[-t2R0:][::-1]) #12 imputed corresponding standard errors
fSEs = np.concatenate((SEs, iSEs)) #12
fSEs = f_fSE(ds, SEs, Vs, t2R0) #12 impute and fill function for SEs
fws = 1/(np.square(fSEs)) #12
adjF = DeltaHat(fws, fds) #12 trim and fill adjusted DeltaHat




#13 data frame of the most relevant arrays
dict_ds = {'ESs': dss, 'Vs': Vss, 'Centered ESs': cen_ds, 'Signed ranks': srl, 'Trimmed centered ESs': tcen_ds, 'Trimmed signed ranks': tsrl,'Trimmed again signed ranks': t2srl,}
dataf = pd.DataFrame(dict_ds)
pd.concat([dataf.head(5), dataf.tail(5)]) # show first and last 5 rows
dataf.to_csv("df.csv")




#14 Fill the funnel plot with the imputed symmetric data
plt.figure(figsize = (12,8))
plt.rcParams.update({'font.size': 19})
plt.scatter(ds, 1/SEs, s = 60, c="r", edgecolor="black", linewidth=1, alpha=0.70)
plt.scatter(ids, 1/iSEs, s = 60, c="g", edgecolor="black", linewidth=1, alpha=0.70)
plt.xlabel("Effect sizes")
plt.ylabel("1/Standard errors")
plt.xlim(0, 1)
plt.ylim(0, 10)
plt.plot([0.35, 0.5], [0, 10], 'k-') # arbitrary line to draw the funnel shape
plt.plot([0.5, 0.65], [10, 0], 'k-') # arbitrary line to draw the funnel shape
plt.plot([F, F], [0, 10], 'r-', linewidth=2) # Original Delta estimate as a red line
plt.plot([adjF, adjF], [0, 10], 'g-', linewidth=2) # Original Delta estimate as a red line
plt.plot([F, F], [0, 10], 'r-', linewidth=2) # Original Delta estimate as a red line
plt.plot([np.mean(ds), np.mean(ds)], [0, 10], 'b-', linewidth=2) # Unweighted mean of original data for illustration
plt.savefig('FP3.png')

