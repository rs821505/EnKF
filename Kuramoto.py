import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *


class KS:
  def __init__(self, domain, L=16, N=128, dt=0.05):
      """
      Parameters
      -------------
      :param L: int,  domain length
      :param N: int,  num degrees freedom
      :param dt: float, timestep
      :param x: np array, domain
      :param k: nd np array, wave numbers
      :param niter: int,     num iterations
      :param E: np array,
      :param E2: np array,
      :param  g:  np array, 
      """
      self.x = domain
      self.dt = dt
      self.k = (np.hstack([np.arange(N//2) ,0,(-1)*np.flip(np.arange(N//2)[1:])]).reshape(-1,1))/L 
      self.E = np.exp(-dt*((self.k**4)-(self.k**2)/2))
      self.E2 = self.E**2
      self.g = -.5j*dt*self.k


  def KS_RHS(self,vin):
      return fft(np.real(ifft(vin))**2)

  def KS_RK4(self,idx, vin):
      
      a = self.g * self.KS_RHS(vin)
      b = self.g * self.KS_RHS(self.E * (vin+a/2))
      c = self.g * self.KS_RHS( self.E * vin + b/2)
      d = self.g * self.KS_RHS(self.E2 * vin + self.E*c)
      vout = self.E2 * vin + ( (self.E2 * a) + (2 * self.E * (b+c)) + d)/6

      return vout

  def advance(self,uin,T):

      niter= int(np.round(T/self.dt))

      vin = fft(uin.T)

      for n in range(niter):

          vin = self.KS_RK4(n,vin)

      vout = vin

      return np.real(ifft(vout))
