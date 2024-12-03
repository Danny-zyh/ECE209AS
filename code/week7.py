import numpy as np
import control as ct


class particle:
    """
    Week 7
    """
    def __init__(self, m=1, qv2=0.1, rv2=0.1, ry2=0.1, rvy=0, f_phi=lambda: 0):
        self.m = m
        self.f_phi = f_phi
        self.qv2 = qv2 # variance of process noise
        self.rv2 = rv2 # variance of observation noise
        self.ry2 = ry2 

        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0], [1 / self.m]])
        self.C = np.eye(2)
        self.Q = np.array([[0, 0], [0, qv2]])
        self.R = np.array([[ry2, rvy], [rvy, rv2]])

    def T(self, s, fi, nq=None):
        if nq is None:
            nq = lambda: np.random.multivariate_normal(np.zeros(2), self.Q)
        
        return self.A @ s + self.B @ np.array([fi]) + nq()
    
    def O(self, s, nr=None):
        if nr is None:
            nr = lambda: np.random.multivariate_normal(np.zeros(2), self.R)
        
        return self.C @ s + nr()
    
    def kf_gain(self):
        K, *_ = ct.dlqe(self.A, np.eye(2), self.C, self.Q, self.R)
        return K
    
    def dynamic_update(self, mu, sigma, fi):
        mu_ = self.A @ mu + self.B @ np.array([fi])
        sigma_ = self.A @ sigma @ self.A.T + self.Q
        return mu_, sigma_

    def measurement_update(self, mu, sigma, K, obs):
        mu_ = mu + K @ (obs - (self.C @ mu))
        sigma_ = (np.eye(2) - K @ self.C) @ sigma
        return mu_, sigma_
