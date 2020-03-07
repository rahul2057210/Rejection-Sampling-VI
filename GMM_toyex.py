import scipy
from sklearn import metrics
#from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
#from matplotlib.pyplot import *
import seaborn as sns
import os,sys
import math
from scipy.special import logsumexp
import autograd
from autograd import grad
import autograd.numpy as np
from autograd.util import quick_grad_check
from scipy.special import gamma
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats as stats



def loss(x):
	mu=x[0]
	sig=np.exp(x[1])

	for m in range(K):
		X=mu+(sig)*Z[:,m]
		P=(stats.norm.pdf(X,2,0.8)+stats.norm.pdf(X,0,0.8)+stats.norm.pdf(X,-2,0.8)+stats.norm.pdf(X,-4,0.8))/4
		logQ=stats.t.logpdf(X,10,mu,sig)
		
		logF=np.log(P[(P>0) & (logQ>-np.inf)])-logQ[(P>0) & (logQ>-np.inf)]
		X=X[(P>0) & (logQ>-np.inf)]
		t=T_value+logF[(~np.isnan(logF))]
		X=X[(~np.isnan(logF))]
		log_accept_prob=np.minimum(0,t)
		log_Z_R=logsumexp(log_accept_prob-np.log(len(t)))   # Sampling distribution is Q
		U=np.random.uniform(0,1,(len(X)))



	gamma_p=np.log(P[np.log(U)<log_accept_prob])
	gamma_r=logQ[np.log(U)<log_accept_prob]   +log_accept_prob[np.log(U)<log_accept_prob]-log_Z_R	

	if alpha<1.0 - 10e-5:
		ratio=(1-alpha)*(gamma_p-gamma_r)

		Max_ratio=max(ratio)

		true_loss= (1/(alpha-1))*((logsumexp(ratio-Max_ratio- np.log(len(ratio))))+Max_ratio )

	elif alpha<1.0 + 10e-5:
		true_loss=np.mean((gamma_r-gamma_p))
	else:
		ratio=(1-alpha)*(gamma_p-gamma_r)

		true_loss= (1/(alpha-1))*((logsumexp(ratio- np.log(len(ratio)))) )


	return (true_loss)


def true_divg(x):
	mu=x[0]
	sig=np.exp(x[1])

	for m in range(K):
		X=mu+(sig)*Z[:,m]
		P=(stats.norm.pdf(X,2,0.8)+stats.norm.pdf(X,0,0.8)+stats.norm.pdf(X,-2,0.8)+stats.norm.pdf(X,-4,0.8))/4
		logQ=stats.t.logpdf(X,10,mu,sig)
		
		logF=np.log(P)-logQ
		t=T_value+logF
		log_accept_prob=np.minimum(0,t)
		log_Z_R=logsumexp(log_accept_prob-np.log(len(t)))   # Sampling distribution is Q

		U=np.random.uniform(0,1,(len(X)))

		Samples=X[np.log(U)<log_accept_prob]      # Sampling distribution is R

		P=(stats.norm.pdf(Samples,2,0.8)+stats.norm.pdf(Samples,0,0.8)+stats.norm.pdf(Samples,-2,0.8)+stats.norm.pdf(Samples,-4,0.8))/4
		Q=stats.t.pdf(Samples,10,mu,sig)
		logF=np.log(P)-np.log(Q)
		t=T_value+logF
		log_accept_prob=np.minimum(0,t)


	gamma_p=np.log(P)
	gamma_r=np.log(Q)+log_accept_prob-log_Z_R


	if alpha<1.0 - 10e-5:
		ratio=(1-alpha)*(gamma_p-gamma_r)

		Max_ratio=max(ratio)

		true_divg= (1/(alpha-1))*((logsumexp(ratio-Max_ratio- np.log(len(ratio))))+Max_ratio )

	elif alpha<1.0 + 10e-5:
		true_divg=np.mean((gamma_r-gamma_p))

	else:
		ratio=(1-alpha)*(gamma_p-gamma_r)

		true_divg= (1/(alpha-1))*((logsumexp(ratio- np.log(len(ratio)))) )



	return (true_divg)


def learned_dist(x):
	mu=x[0]
	sig=np.exp(x[1])

	for m in range(K):
		X=X1
		P=(stats.norm.pdf(X,2,0.8)+stats.norm.pdf(X,0,0.8)+stats.norm.pdf(X,-2,0.8)+stats.norm.pdf(X,-4,0.8))/4
		logQ=stats.t.logpdf(X,10,mu,sig)	
		logF=np.log(P)-logQ
		t=T_value+logF
	
		log_accept_prob=np.minimum(0,t)
		log_Z_R=logsumexp(log_accept_prob-np.log(len(t)))   # Sampling distribution is Q


	
	gamma_r=logQ+log_accept_prob-log_Z_R

	return (np.exp(gamma_r))

def T_val(x,gamma):
	mu=x[0]
	sig=np.exp(x[1])
	for m in range(K):
		X=mu+(sig)*Z[:,m]
		P=(stats.norm.pdf(X,2,0.8)+stats.norm.pdf(X,0,0.8)+stats.norm.pdf(X,-2,0.8)+stats.norm.pdf(X,-4,0.8))/4
		logQ=stats.t.logpdf(X,10,mu,sig)
		
		logF=np.log(P[(P>0) & (logQ>-np.inf)])-logQ[(P>0) & (logQ>-np.inf)]
		T_value=np.quantile(-logF[(~np.isnan(logF))],gamma,interpolation="lower")-np.log(2)
		
	return (T_value)

def true_sample():
	U=np.random.uniform(0,1,1)
	if U<1/4:
		x=np.random.normal(-4,0.8,1)
	elif U>=(1/4) and U<(2/4):
		x=np.random.normal(-2,0.8,1)
	elif U>=(2/4) and U<(3/4):
		x=np.random.normal(0,0.8,1)
	else:
		x=np.random.normal(2,0.8,1)

	return x



#### Generating data
S=int(sys.argv[1])
K=int(sys.argv[2])
Num_sample=int(sys.argv[3])


Z=np.random.standard_t(10,(S,K))


count=np.array([1.0 ])#,0.8,0.6,0.5])
Result=np.zeros((3,len(count)))
index=0


new_grad=grad(loss)

T_value=np.inf

for alpha in count:
	x0=np.array([0.01,-10])
	param=x0

	m1 = 0
	m2 = 0
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	t = 0
	learning_rate =0.1

	for epoch in range(1000):

		t += 1

		if (epoch+1)%10==0:
			T_value=T_val(param,0.7)

		gradient=new_grad(param)

		m1 = beta1 * m1 + (1 - beta1) * gradient
		m2 = beta2 * m2 + (1 - beta2) * gradient**2
		m1_hat = m1 / (1 - beta1**t)
		m2_hat = m2 / (1 - beta2**t)
		param -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)

		if epoch%200==0:
			learning_rate=max(learning_rate/2.0,0.001)

	x0=param

	Accep=[]
	T_value	= T_val(param,0.7)
	logM_learned=-T_val(param,0.7)

	for j in range(Num_sample):
		U=np.random.uniform(0,1,1)
		x=np.random.standard_t(10,1)
		#x=x0[0]+np.exp(x0[1])*x
		x=true_sample()
		p=(stats.norm.pdf(x,2,0.8)+stats.norm.pdf(x,0,0.8)+stats.norm.pdf(x,-2,0.8)+stats.norm.pdf(x,-4,0.8))/4
		t=np.log(p)-logM_learned-stats.t.logpdf(x,10,x0[0],np.exp(x0[1]))
		t=np.minimum(0,t)
	
		if (np.log(U)<(t)):
			Accep.append(x)


	print('D_(q||p)',loss(x0))
	print('D_(r||p)',true_divg(x0))
	print('Acceptance',len(Accep)/Num_sample)

	Result[0,index]=loss(x0)
	Result[1,index]=true_divg(x0)
	Result[2,index]=len(Accep)/Num_sample

	index=index+1

X1=np.arange(-7,7,0.01)

p1=plt.scatter(X1,stats.t.pdf(X1,10,x0[0],np.exp(x0[1])),label="t_"+str(10))

np.savetxt('GMM_Results',Result)

X1=np.arange(-7,5,0.01)

p2=plt.scatter(X1,(stats.norm.pdf(X1,2,0.8)+stats.norm.pdf(X1,0,0.8)+stats.norm.pdf(X1,-2,0.8)+stats.norm.pdf(X1,-4,0.8))/4,label="real",color='red')

p3=plt.scatter(X1,learned_dist(x0),label="learn",color='black')


plt.legend( fontsize = 'xx-large')
plt.savefig('GMM_toy_ex.png')


