"""==========================================================================================
Ensemble Kalman Filter (EnKF) for Lorenz 96 (L96). See Law et al. (2015):

"Data Assimilation A Mathematical Introduction" https://doi.org/10.48550/arXiv.1506.07825
============================================================================================="""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy.matlib
import numpy.linalg as LA
from numpy.linalg import inv


class EnKF:
  def __init__(self, fmodel, localization=True, inf_factor=2, num_steps=1000, num_samps=50, tau=1.5):
    """
    Parameters:
    -----------
    :param localization: bool, choice to employ localization
    :param inf_factor:   int, inflaction factor
    :param n_dim:        int, dimension of state variable, L63 is 3D problem
    :param n_steps:      int, number of assimilation steps 
    :param n_samps:      int, ensemble size
    :param T:            float, observation time interval
    :param oVar:         float,  observation error variance
    :param H:            nd np array, observation matrix
    :param n_obs:        int, dimension of single observation
    :param tau:          float, stopping criteria constant
    :param utrues:       nd np array, n_steps x n_dim time series of true values
    :param vs:           nd np array, n_steps x n_obs time series of observation data
    :param uests:        nd np array, n_steps x n_dim time series of the PF post estimates
    :param: error_stats: nd np array, n_steps x 3     time series of RMSE, L2, Misfit Error
    :param L:            nd np array, n_dim x n_dim localization matrix
    :param Cpr:          nd np array, n_dim x n_dim  prior covariance from the ensemble
    :param uens:         nd np array, n_samps x n_dim  mean state vector
    :param v:            np array,    n_obs/4 x 1 assimilated observations
    :param pertv:        nd np array, n_samps x n_obs/4 perturbed observations
    :param gamma:        nd np array, n_samps x n_dim observational noise
    :param fmodel:       object,  forward model simulation object
    """
    self.localization = localization
    self.inf_factor = inf_factor
    self.n_dim = 40         
    self.n_steps = num_steps     
    self.n_samps = num_samps   
    self.T = .1                
    self.oVar = .5         
    self.H = np.eye(self.n_dim)  
    self.H = self.H[::4,:]
    self.n_obs = self.H.shape[0]
    self.tau = tau
    self.Cpr = None
    self.utrues = np.zeros((self.n_steps, self.n_dim))  
    self.vs  = np.zeros((self.n_steps, self.n_obs))     
    self.uests = np.zeros((self.n_steps, self.n_dim))   
    self.error_stats = np.zeros((self.n_steps, 3)) 
    self.L = np.zeros([self.n_dim,self.n_dim])
    self.uens = np.zeros([self.n_samps,self.n_dim])
    self.v  = np.zeros([self.n_obs//4])
    self.pertv = np.zeros([self.n_samps,self.n_obs//4])
    self.eta = np.zeros([self.n_samps, self.n_obs])
    self.gamma = np.random.randn(self.n_samps, self.n_dim) 
    self.f_model= fmodel
    
  # Seed for random numbers
  np.random.seed(0) 

  def Localization(self):
    """
    Create Localization matrix
    """
    q = np.arange(0,self.n_dim)
    loc_w = (np.cos((q-1)/self.n_dim*np.pi*2)+1)/2
    
    for i in range(self.n_dim):
        self.L[i,:] = np.roll(loc_w,(i-1))
        

  def Init_Ensemble(self):
    """
    Create localizaion Matrix if desired
    Initial value for L63
    Generate true signal using advanceL63
    Solve forward model (L96) for duration of T
    return: Initialized particles 
    """

    if self.localization:
      self.Localization()

    uin = np.random.randn(1, self.n_dim)     

    for i in range(self.n_steps):

      self.utrues[i, :] = self.f_model.advanceL96(uin, self.T) 
      uin = self.utrues[i, :]

    uprior = self.utrues[1, :] 
    test = np.matlib.repmat(uprior,self.n_samps, 1)+np.sqrt(self.oVar)*self.gamma

    return np.matlib.repmat(uprior,self.n_samps, 1)+np.sqrt(self.oVar)*self.gamma

  def Run_Forward(self):
    """
    Run the forward model
    """
    for j in range(self.n_samps):
        self.uens[j, :] = self.f_model.advanceL96(self.uens[j, :], self.T)

  def Forecast_Step(self,step):
    """
    Run forward model
    Assimilate observations v
    Save observation for post-processing
    return: Assimilated observations, Prior covariance matrix
    """
    self.v = np.dot(self.H,self.utrues[step, :][np.newaxis].T)+ np.sqrt(self.oVar)*np.random.randn(self.n_obs, 1)
    self.vs[step, :] = self.v.ravel() 

    if self.localization:
        return np.cov(self.uens.T)*self.L

    return np.cov(self.uens.T)


  def Update_Step(self):
    """
    Compute Kalman gain and noise 
    Perturb Observations
    return: updated mean state vector
    """
    
    K = self.Cpr.dot(self.H.T).dot(inv(self.H.dot(self.Cpr).dot(self.H.T)+self.oVar*np.eye(self.n_obs))) 
    self.eta = np.random.randn(self.n_samps, self.n_obs)
    self.pertv = np.matlib.repmat(self.v.T,self.n_samps, 1) + self.inf_factor*np.sqrt(self.oVar)*self.eta


    return self.uens + (self.pertv - self.uens.dot(self.H.T)).dot(K.T)          


  def Run_Ensemble(self):
    """
    Initialize ensemble
    Prediction/Forecast Step
    Analysis/Update Step
    Save the posterior mean, rms,l2, and misfit error
    """

    self.uens = self.Init_Ensemble()

    for i in range(self.n_steps):

      self.Run_Forward()
      self.Cpr = self.Forecast_Step(i)
      self.uens = self.Update_Step()
        
      self.uests[i, :] = self.uens.mean(axis=0)
      self.error_stats[i, 0] = np.sqrt(np.mean((self.uests[i,:]-self.utrues[i,:])**2))
      self.error_stats[i, 1] = LA.norm(self.uests[i,:]-self.utrues[i,:])
      self.error_stats[i, 2] = LA.norm((self.pertv - self.uens.dot(self.H.T)))
      
      # Stopping criteria
      # if self.misfit[i] <= self.tau*LA.norm(self.eta):
      #   break
  

  def EnKF_Plot(self):
    """
    Plot long term statistics of ensemble kalman filter

    Plot true and estimated state values 
    Plot time averaged error statistics
    """

    dim_plots=3; stat_plots=3
    xlabels= ['Time', 'Assimilation Step']
    ylabels = ['$1^{st}$ Component','$2^{nd}$ Component','$3^{rd}$ Component',
               'RMSE',"$L^2$ Error","Data Misfit" ]
  
    f, axarr = plt.subplots(dim_plots,figsize=(18,18))

    for index in range(dim_plots):

      axarr[index].plot(self.utrues[:,index+1])
      axarr[index].plot(self.uests[:,index+1],'k--')
      axarr[index].set_xlabel(xlabels[0],fontsize=14,color='red')
      axarr[index].set_ylabel(ylabels[index],fontsize=14,color='red')
      axarr[index].set_xlim([0,self.n_steps])
      axarr[index].legend(['true','post'],loc='upper left')
      axarr[index].grid()
      plt.tight_layout()

    g, axarr1 = plt.subplots(stat_plots,figsize=(18,18))

    for index in range(stat_plots):
 
      axarr1[index].plot(self.error_stats[:,index])
      axarr1[index].set_title('Ensemble Kalman Filter for L63',fontsize=14,color='red')
      axarr1[index].set_xlabel(xlabels[1],fontsize=14,color='red')
      axarr1[index].set_ylabel(ylabels[index+3],fontsize=14,color='red')
      axarr1[index].set_xlim([0,self.n_steps])
      axarr1[index].grid()
      plt.tight_layout()

  def Metrics(self):
    print('Time Averaged RMSE: \n',self.error_stats[:,0].mean(),'\n')
    print('Minimum RMSE: \n', self.error_stats[:,0].min(),'\n')
    print('Time Averaged L2 Error: \n', self.error_stats[:,1].mean(),'\n')
    print('Minimum L2 Error: \n', self.error_stats[:,1].min(),'\n')
    print('Time Averaged Misfit: \n', self.error_stats[:,2].mean(),'\n')
    print('Minimum misfit: \n', self.error_stats[:,2].min(),'\n')
    # print('Ieration Number Min RMSE:',np.where(self.rmses==self.rmses.min()),'\n')
