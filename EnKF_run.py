from Lorenz96 import L96
from EnsembleKF import EnKF

if __name__== "__main__":

  Forward_model = L96()
  Filter = EnKF(Forward_model,num_steps=500)
  Filter.Run_Ensemble()
  Filter.EnKF_Plot()
  Filter.Metrics()
