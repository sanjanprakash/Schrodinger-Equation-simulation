import numpy as np
from scipy import fftpack

#Class to obtain a solution of the Schrodinger equation for a given potential
class Schrodinger(object):
	
	#All parameters of the class are validated and initialised here
	def __init__(self, x, psi_x0, V_x, k0 = None, hbar = 1, m = 1, t0 = 0.0):

		#Validation of array inputs
		self.x, psi_x0, self.V_x = map(np.asarray, (x, psi_x0, V_x))
		N = self.x.size

		#Set internal parameters
		assert hbar > 0
		assert m > 0
		self.hbar = hbar
		self.m = m
		self.t = t0
		self.dt_ = None
		self.N = len(x)
		self.dx = self.x[1] - self.x[0]
		self.dk = 2 * np.pi / (self.N * self.dx)

		#Set momentum scale
		if k0 == None:
			self.k0 = -0.5 * self.N * self.dk
		else:
			assert k0 < 0
			self.k0 = k0
		self.k = self.k0 + self.dk * np.arange(self.N)

		self.psi_x = psi_x0
		self.compute_k_from_x()

		#Variables which hold steps in evolution
		self.x_evolve_half = None
		self.x_evolve = None
		self.k_evolve = None

	def _set_psi_x(self, psi_x):
		assert psi_x.shape == self.x.shape
		self.psi_mod_x = (psi_x * np.exp(-1j * self.k[0] * self.x) * self.dx / np.sqrt(2 * np.pi))
		self.psi_mod_x /= self.norm
		self.compute_k_from_x()

	def _get_psi_x(self):
		return (self.psi_mod_x * np.exp(1j * self.k[0] * self.x) * np.sqrt(2 * np.pi) / self.dx)

	def _set_psi_k(self, psi_k):
		assert psi_k.shape == self.x.shape
		self.psi_mod_k = psi_k * np.exp(1j * self.x[0] * self.dk * np.arange(self.N))
		self.compute_x_from_k()
		self.compute_k_from_x()

	def _get_psi_k(self):
		return self.psi_mod_k * np.exp(-1j * self.x[0] * self.dk * np.arange(self.N))

	def _get_dt(self):
		return self.dt_

	def _set_dt(self, dt):
		assert dt != 0
		if dt != self.dt_:
			self.dt_ = dt
			self.x_evolve_half = np.exp(-0.5 * 1j * self.V_x / self.hbar * self.dt)
			self.x_evolve = self.x_evolve_half * self.x_evolve_half
			self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * (self.k * self.k) * self.dt)

	def _get_norm(self):
		return self.wf_norm(self.psi_mod_x)

	psi_x = property(_get_psi_x, _set_psi_x)
	psi_k = property(_get_psi_k, _set_psi_k)
	norm = property(_get_norm)
	dt = property(_get_dt, _set_dt)

	#The Fourier transform
	def compute_k_from_x(self):
		self.psi_mod_k = fftpack.fft(self.psi_mod_x)
	#The inverse Fourier transform
	def compute_x_from_k(self):
		self.psi_mod_x = fftpack.ifft(self.psi_mod_k)

	#To calculate the norm of a wave function
	def wf_norm(self, wave_fn):
		assert wave_fn.shape == self.x.shape
		return np.sqrt((abs(wave_fn) ** 2).sum() * 2 * np.pi / self.dx)

	def solve(self, dt, Nsteps = 1, eps = 1e-3, max_iter = 1000):
		eps = abs(eps)
		assert eps > 0
		t0 = self.t
		old_psi = self.psi_x
		d_psi = 2 * eps
		num_iter = 0
		while (d_psi > eps) and (num_iter <= max_iter):
			num_iter += 1
			self.time_step(-1j * dt, Nsteps)
			d_psi = self.wf_norm(self.psi_x - old_psi)
			old_psi = 1. * self.psi_x
		self.t = t0

	#Discretization and solving for each time interval...
	def time_step(self, dt, Nsteps = 1):
		assert Nsteps >= 0
		self.dt = dt
		if Nsteps > 0:
			self.psi_mod_x *= self.x_evolve_half
			for num_iter in xrange(Nsteps - 1):
				self.compute_k_from_x()
				self.psi_mod_k *= self.k_evolve
				self.compute_x_from_k()
				self.psi_mod_x *= self.x_evolve
			self.compute_k_from_x()
			self.psi_mod_k *= self.k_evolve
			self.compute_x_from_k()
			self.psi_mod_x *= self.x_evolve_half
			self.compute_k_from_x()
			self.psi_mod_x /= self.norm
			self.compute_k_from_x()
			self.t += dt * Nsteps