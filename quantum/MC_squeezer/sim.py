import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import warnings
from numpy import *

class MCSqueezer():
    '''Stands for Monte Carlo squeezer, in reference to MC Hammer, who reminds us that we cannot touch this without
    measurement dephasing.
    The basic idea is, if we can represent states of light as vectors in a
    real or imaginary vector state, and write down operations such as squeezing or loss a matrix operators, then we
    can visualize squeezing experiments by, first, sampling many points in the state space and, second, performing the
    matrix operations on that ensemble. Hence the Monte Carlo name.'''
    def __init__(self, N, sigma, use_cp=True):

        self.N = N
        self.use_cp = use_cp
        self.samples_history = []

        self.sigma = sigma

        self.step_names = []

        if use_cp == True:
            try:
                import cupy
                self.lib = cupy
            except:
                warnings.warn("Cupy installation not found. Using numpy.", RuntimeWarning)
                self.lib = np
        else:
            self.lib = np

        self.samples = self.gen_vac_noise(self.sigma)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Vacuum noise')

    def gen_vac_noise(self, sigma, testpoints=False):
        '''
        Creates a vacuum noise distribution (for initial states or loss), isotropic in the 4D space.
        '''
        lib = self.lib

        samples = lib.random.normal(0, sigma, size=(4, self.N))  # create random samples

        if testpoints == True:
            samples[:, 0] = lib.array([sigma, 0, 0, 0])
            samples[:, 1] = lib.array([0, sigma, 0, 0])
            samples[:, 2] = lib.array([0, 0, sigma, 0])
            samples[:, 3] = lib.array([0, 0, 0, sigma])

        return samples

    def hotloss(self, eta, sigma):
        '''
        An operation that effectively acts as a beamsplitter between both signal and idler and noise (which may be hot, hence hotloss)
        Models any system loss/coupling to environment
        '''

        lib = self.lib

        noise = self.gen_vac_noise(sigma)

        self.samples = self.samples * lib.sqrt(eta) + lib.sqrt(1 - eta) * noise

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Loss')

    def addnoise(self, sigma):
        '''
        Models a HEMT (non-quantum object)
        '''

        lib = self.lib

        noise = self.gen_vac_noise(sigma)

        self.samples = self.samples + noise

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Added Noise')

    def propeller(self, kerr):

        N = self.N
        lib = self.lib

        alpha = lib.sqrt(lib.sum(self.samples[0:2, :]**2, axis=0))
        beta = lib.sqrt(lib.sum(self.samples[2:, :]**2, axis=0))

        theta_a = alpha * kerr
        theta_b = beta * kerr

        if self.use_cp:
            theta_a = lib.asnumpy(theta_a)   # there's a bug or something in cupy that prevents you from using both cupy arrays and numerical literals together when defining an array
            theta_b = lib.asnumpy(theta_b)

        op = lib.array([[np.cos(theta_a), -np.sin(theta_a), np.zeros(N), np.zeros(N)],
                         [np.sin(theta_a), np.cos(theta_a), np.zeros(N), np.zeros(N)],
                         [np.zeros(N), np.zeros(N), np.cos(theta_b), -np.sin(theta_b)],
                         [np.zeros(N), np.zeros(N), np.sin(theta_b), np.cos(theta_b)]])

        product = op*self.samples

        '''
        tried to find a numpy function that does this vectorized matrix multiplication, but couldn't find what I needed
        and made this implementation
        '''
        self.samples = lib.sum(product, axis=1)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Propeller')

    def four_wave_selfkerr(self, kerr, time, steps):

        dt = time / (steps - 1)

        N = self.N
        lib = self.lib

        for i in range(0, steps - 1):
            print(i)
            alpha = lib.sqrt(lib.sum(self.samples[0:2, :] ** 2, axis=0))
            beta = lib.sqrt(lib.sum(self.samples[2:, :] ** 2, axis=0))

            phi = lib.zeros(N)

            r_a = 0.5 * kerr * alpha ** 2 * dt
            r_b = 0.5 * kerr * alpha ** 2 * dt

            theta_a = -2 * alpha ** 2 * kerr * dt
            theta_b = -2 * beta ** 2 * kerr * dt

            scale_a = lib.ones(N) + kerr * alpha ** 2 * dt
            scale_b = lib.ones(N) + kerr * beta ** 2 * dt

            I = lib.array(
                [[lib.ones(N), lib.zeros(N), lib.zeros(N), lib.zeros(N)],
                 [lib.zeros(N), lib.ones(N), lib.zeros(N), lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), lib.ones(N), lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), lib.zeros(N), lib.ones(N)]])

            S = lib.array(
                [[lib.cosh(r_a) + lib.cos(phi) * lib.sinh(r_a), lib.sin(phi) * lib.sinh(r_a), lib.zeros(N),
                  lib.zeros(N)],
                 [lib.sin(phi) * lib.sinh(r_a), lib.cosh(r_a) - lib.sinh(r_a) * lib.cos(phi), lib.zeros(N),
                  lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), lib.cosh(r_b) + lib.cos(phi) * lib.sinh(r_b),
                  lib.sin(phi) * lib.sinh(r_b)],
                 [lib.zeros(N), lib.zeros(N), np.sin(phi) * np.sinh(r_b),
                  lib.cosh(r_b) - lib.sinh(r_b) * lib.cos(phi)]])

            D = lib.array(
                [[scale_a, lib.zeros(N), lib.zeros(N), lib.zeros(N)],
                 [lib.zeros(N), scale_a, lib.zeros(N), lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), scale_b, lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), lib.zeros(N), scale_b]])

            R = lib.array([[lib.cos(theta_a), -lib.sin(theta_a), lib.zeros(N), lib.zeros(N)],
                           [lib.sin(theta_a), lib.cos(theta_a), lib.zeros(N), lib.zeros(N)],
                           [lib.zeros(N), lib.zeros(N), lib.cos(theta_b), -lib.sin(theta_b)],
                           [lib.zeros(N), lib.zeros(N), lib.sin(theta_b), lib.cos(theta_b)]])

            productR = R * self.samples
            samples1 = lib.sum(productR, axis=1)
            productD = D * samples1
            samples2 = lib.sum(productD, axis=1)
            productS = S * samples2
            self.samples = lib.sum(productS, axis=1)
            self.samples = lib.clip(self.samples, a_min=-1000, a_max=1000)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Self Kerr 4 Wave')

    def four_wave_crosskerr(self, kerr, time, steps):

        dt = time/(steps-1)

        N = self.N
        lib = self.lib

        for i in range(0,steps-1):
            print(i)
            sI = self.samples[0, :]
            sQ = self.samples[1, :]
            iI = self.samples[2, :]
            iQ = self.samples[3, :]

            r_a = 0.5*kerr * alpha ** 2 * dt
            r_b = 0.5*kerr * alpha ** 2 * dt

            theta_a = -2 * alpha ** 2 * kerr*dt
            theta_b = -2 * beta ** 2 * kerr*dt

            scale_a = lib.ones(N) + kerr * alpha ** 2 * dt
            scale_b = lib.ones(N) + kerr * beta ** 2 * dt

            I = lib.array(
                [[lib.ones(N), lib.zeros(N), lib.zeros(N), lib.zeros(N)],
                 [lib.zeros(N), lib.ones(N), lib.zeros(N), lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), lib.ones(N), lib.zeros(N)],
                 [lib.zeros(N), lib.zeros(N), lib.zeros(N), lib.ones(N)]])

            S = lib.array([[lib.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                            [0, np.cosh(r), np.sin(phi) * np.sinh(r), np.cos(phi) * np.sinh(r)],
                            [np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), np.cosh(r), 0],
                            [-np.sin(phi) * np.sinh(r), np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])

            B = lib.array([[lib.cos(phi), lib.zeros(N), lib.sin(phi), lib.zeros(N)],
                            [lib.zeros(N), lib.cos(phi), lib.zeros(N), lib.sin(phi)],
                            [-lib.sin(phi), lib.zeros(N), lib.cos(phi), lib.zeros(N)],
                            [lib.zeros(N), - lib.sin(phi), lib.zeros(N), lib.cos(phi)]])

            print(self.samples)
            productR = R * self.samples
            samples1 = lib.sum(productR, axis=1)
            print(samples1)
            productD = D * samples1
            samples2 = lib.sum(productD, axis=1)
            print(samples2)
            productS = S * samples2
            self.samples = lib.sum(productS, axis=1)
            print(self.samples)


        self.samples_history.append(self.samples.copy())

        self.step_names.append('Cross Kerr 4 Wave')

    def one_mode_squeeze(self, phi, r):

        lib = self.lib

        op = lib.array([[np.cosh(r) + np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), 0, 0],
                          [np.sin(phi) * np.sinh(r), np.cosh(r) - np.sinh(r) * np.cos(phi), 0, 0],
                          [0, 0, np.cosh(r) + np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r)],
                          [0, 0, np.sin(phi) * np.sinh(r), np.cosh(r) - np.sinh(r) * np.cos(phi)]])

        self.samples = lib.dot(op, self.samples)

        self.samples = lib.clip(self.samples, a_min=-1000, a_max=1000)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('1MS')

    def one_mode_squeeze_signal(self, phi, r):
            lib = self.lib

            op = lib.array([[np.cosh(r) + np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), 0, 0],
                            [np.sin(phi) * np.sinh(r), np.cosh(r) - np.sinh(r) * np.cos(phi), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

            self.samples = lib.dot(op, self.samples)

            self.samples = lib.clip(self.samples, a_min=-1000, a_max=1000)

            self.samples_history.append(self.samples.copy())

            self.step_names.append('1MS')

    def two_mode_squeeze(self, phi, r):

        lib = self.lib

        op = lib.array([[np.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                        [0, np.cosh(r), np.sin(phi) * np.sinh(r), np.cos(phi) * np.sinh(r)],
                        [np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), np.cosh(r), 0],
                        [-np.sin(phi) * np.sinh(r), np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])

        self.samples = lib.dot(op, self.samples)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('2MS')

    def beamsplit_signal_idler(self, theta, phi):

        lib = self.lib

        op = lib.array([[np.cos(theta),              0,                          np.sin(theta)*np.cos(phi),  np.sin(theta)*np.sin(phi)],
                        [0,                          np.cos(theta),              -np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi)],
                        [-np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),  np.cos(theta),              0],
                        [-np.sin(theta)*np.sin(phi), -np.sin(theta)*np.cos(phi), 0,                          np.cos(theta)]])

        self.samples = lib.dot(op, self.samples)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Passive Beamsplitter')

    def conditionally_rotate_signal(self, theta):

        lib = self.lib

        op1 = lib.array([[np.cos(-theta / 2), -np.sin(-theta / 2), 0, 0],
                         [np.sin(-theta / 2), np.cos(-theta / 2), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        op2 = lib.array([[np.cos(theta / 2), -np.sin(theta / 2), 0, 0],
                         [np.sin(theta / 2), np.cos(theta / 2), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        self.samples[:, 0:self.N // 2] = lib.dot(op1, self.samples[:, 0:self.N // 2])
        self.samples[:, self.N // 2:] = lib.dot(op2, self.samples[:, self.N // 2:])

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Disp. Shift')

    def conditionally_displace_signal(self, theta, r):

        lib = self.lib

        self.samples[:, 0:self.N // 2] = self.samples[:, 0:self.N // 2] + lib.expand_dims(lib.array([np.cos(theta)*r, np.sin(theta)*r, 0, 0]), 1)
        self.samples[:, self.N // 2:] = self.samples[:, self.N // 2:] - lib.expand_dims(lib.array([np.cos(theta)*r, np.sin(theta)*r, 0, 0]), 1)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Cond. Displacement')

    def displace(self, dsi, dsq, dii, diq):

        lib = self.lib

        self.samples = self.samples + lib.expand_dims(lib.array([dsi, dsq, dii, diq]), 1)

        self.samples_history.append(self.samples.copy())

        self.step_names.append('Displace')


