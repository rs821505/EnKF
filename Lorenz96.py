"""=================================================================================
 Lorenz 96 (L96). See:

"https://en.wikipedia.org/wiki/Lorenz_system for Lorenz 63 (L63)
===================================================================================="""

# Import Modules
import numpy as np

class L96:
  def __init__(self,F=8, dt=0.01):
    """
    :param dt: float, timestep
    :param F: int, forcing constant
    """

    self.dt = dt
    self.F = F

  def RHSL96(self,uin):
    return (np.roll(uin,-1)-np.roll(uin,2))*np.roll(uin,1)-uin+self.F

  def RK4L96(self,uin):
      k1 = self.dt*self.RHSL96(uin)
      k2 = self.dt*self.RHSL96(uin+k1/2)
      k3 = self.dt*self.RHSL96(uin+k2/2)
      k4 = self.dt*self.RHSL96(uin+k3)
      uout = uin + (k1+2*k2+2*k3+k4)/6
      return uout

  def advanceL96(self,uin,T):

      niter = int(T/self.dt)

      for i in range(niter):
          uin = self.RK4L96(uin)

      uout = uin
      return uout
