from tqdm import tqdm
import numpy as np
import project1
import helpers
import matplotlib.pyplot as plt

def main():
	p = helpers.Simple1()
	(x_hist_1, f_hist_1) = run_optimizer(project1.gradient_descent, p, [0.2, 2])
	p = helpers.Simple2()
	(x_hist_2, f_hist_2) = run_optimizer(project1.gradient_descent, p, [-2, 1.7])
	p = helpers.Simple3()
	(x_hist_3, f_hist_3) = run_optimizer(project1.gradient_descent, p, [1,1,1,1])


	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	ax2.plot(f_hist_1)
	ax2.set_title('Simple1 Convergence Plot')
	ax2.set_xlabel('Iterations')
	ax2.set_ylabel('Value')

	ax3.plot(f_hist_2)
	ax3.set_title('Simple2 Convergence Plot')
	ax3.set_xlabel('Iterations')
	ax3.set_ylabel('Value')

	ax4.plot(f_hist_3)
	ax4.set_title('Simple3 Convergence Plot')
	ax4.set_xlabel('Iterations')
	ax4.set_ylabel('Value')

	
	contourSize = [50,50]
	X = np.linspace(-2,2,contourSize[0])
	Y = np.linspace(-1,2,contourSize[1])
	Z = np.zeros(contourSize)
	p = helpers.Simple1()
	for i in range(len(X)):
		for j in range(len(Y)):
			Z[j][i] = p.f([X[i],Y[j]])
	ax1.contour(X,Y,Z, 100)
	x_hist_1 = np.array(x_hist_1)
	ax1.plot(x_hist_1[:,0],x_hist_1[:,1])
	p = helpers.Simple1()
	(x_hist_1b, f_hist_1b) = run_optimizer(project1.gradient_descent, p, [-1, 2])
	p = helpers.Simple1()
	(x_hist_1c, f_hist_1c) = run_optimizer(project1.gradient_descent, p, [1, 0])
	x_hist_1b = np.array(x_hist_1b)
	ax1.plot(x_hist_1b[:,0],x_hist_1b[:,1])
	x_hist_1c = np.array(x_hist_1c)
	ax1.plot(x_hist_1c[:,0],x_hist_1c[:,1])
	ax1.set_title('Simple1 Contour Plot')

	plt.tight_layout()

	plt.show()


def run_optimizer(optimizer, p, x0):
    x_hist = optimizer(p.f, p.g, x0, p.n, p.count, p.prob)
    f_hist = [p.f(j) for j in x_hist]
    return (x_hist, f_hist)

if __name__ == '__main__':
	main()