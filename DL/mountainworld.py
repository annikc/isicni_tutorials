
import numpy as np

class MountainWorld(object):
	
	def __init__(self):
		self.pos = -np.pi/6
		self.vel = 0.0

	def init(self):
		self.pos = -0.6 + 0.2*np.random.rand()
		self.vel = 0.0

	def get_state(self):
		return (self.pos, self.vel)

	def set_state(self, pos, vel):
		self.pos = pos
		self.vel = vel

	def bound_pos(self,pos):
		return max(min(pos,0.6),-1.2)

	def bound_vel(self,vel):
		return max(min(vel,0.07),-0.07)

	def move(self,action):
		if self.pos <= -1.2:
			self.vel = 0.0
		self.vel = self.bound_vel(self.vel + 0.001*action - 0.0025*np.cos(3*self.pos))
		self.pos = self.bound_pos(self.pos + self.vel)
		is_terminal = True if self.pos >= 0.6 else False
		reward = -1 if self.pos < 0.5 else 0
		return ((self.pos,self.vel), reward, is_terminal)