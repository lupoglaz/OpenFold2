
##Rewritten https://github.com/pmocz/nbody-python
##problem: when two particles are close energy is not conserved
import sys
import numpy as np
import _pickle as pkl
import argparse
class SimConfig:
	N=100    # Number of particles
	tEnd=0.5   # time at which simulation ends
	dt=0.01   # timestep
	G=1.0    # Gravitational Constant
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

def get_acc(pos, mass, G, eps=0.1):
	x, y, z = pos[:,0:1], pos[:,1:2], pos[:,2:]
	dx, dy, dz = x.T - x, y.T - y, z.T - z
	inv_r3 = (dx**2 + dy**2 + dz**2 + eps**2)**(-1.5)
	ax, ay, az = G*(dx*inv_r3)@mass, G*(dy*inv_r3)@mass, G*(dz*inv_r3)@mass
	return np.hstack((ax,ay,az))

def getEnergy( pos, vel, mass, G, eps=1E-9):
	KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
	x, y, z = pos[:,0:1], pos[:,1:2], pos[:,2:]
	dx, dy, dz = x.T - x, y.T - y, z.T - z
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	return KE, PE

def simulate(mass, pos, vel, config):
	Nt = int(np.ceil(config.tEnd/config.dt))

	positions = np.zeros((Nt, pos.shape[0], pos.shape[1]))
	velocities = np.zeros((Nt, vel.shape[0], vel.shape[1]))
	
	t = 0.0
	acc = get_acc(pos, mass, config.G)
	for i in range(Nt):
		vel += acc * config.dt/2.0
		pos += vel * config.dt
		acc = get_acc(pos, mass, config.G)
		vel += acc * config.dt/2.0
		t += config.dt

		positions[i,:,:] = pos.copy()
		velocities[i,:,:] = vel.copy()
	
	return positions, velocities

def visualize(r, v, config):
	import matplotlib.pylab as plt
	from mpl_toolkits.mplot3d import Axes3D
	import seaborn as sea
	sea.set_style("whitegrid")
	from matplotlib import pylab as plt
	from celluloid import Camera

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	camera = Camera(fig)
	KE = []
	PE = []
	TE = []
	for i in range(r.shape[0]):
		pos, vel = r[i,:,:], v[i,:,:]
		KEt, PEt = getEnergy(pos, vel, mass, config.G)
		KE.append(KEt)
		PE.append(PEt)
		TE.append(KEt + PEt)
		ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=20, c="r")
		camera.snap()
	
	ax.set_xlim(-2,2)
	ax.set_ylim(-2,2)
	ax.set_zlim(-2,2)
	animation = camera.animate()
	animation.save('anim.gif')

	fig = plt.figure(figsize=(12,6))
	plt.plot(KE, label='Kinetic energy')
	plt.plot(PE, label='Potential energy')
	plt.plot(TE, label='Total energy')
	plt.legend()
	plt.savefig('energy.png')

def generate_train_set(num_sim=100):
	sims = []
	for i in range(num_sim):
		N = np.random.randint(5, 20)
		mass = 20.0*np.ones((N,1))/N
		pos  = np.random.randn(N,3)
		vel  = np.random.randn(N,3)
		vel -= np.mean(mass * vel, 0) / np.mean(mass)
		config = SimConfig(tEnd = 2.0)
		
		r, v = simulate(mass, pos, vel, config)
		sims.append((r,v))

	with open('data_train.pkl', 'wb') as fout:
		pkl.dump(sims, fout)
	# visualize(r, v, config)

def generate_test_set():
	N = np.random.randint(5, 20)
	mass = 20.0*np.ones((N,1))/N
	pos  = np.random.randn(N,3)
	vel  = np.random.randn(N,3)
	vel -= np.mean(mass * vel, 0) / np.mean(mass)
	config = SimConfig(tEnd = 2.0)
	r, v = simulate(mass, pos, vel, config)
	sims = [(r,v)]
	with open('data_test.pkl', 'wb') as fout:
		pkl.dump(sims, fout)
	
	# visualize(r, v, config)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')
	args = parser.parse_args()

	if args.cmd is None:
		parser.print_help()
		sys.exit()
	elif args.cmd() == 'test':
		generate_test_set()
	elif args.cmd() == 'train':
		generate_train_set()
	