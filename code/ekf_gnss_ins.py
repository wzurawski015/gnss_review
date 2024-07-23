# projekt/kod/ekf_gnss_ins.py
# project/code/ekf_gnss_ins.py

import numpy as np

class EKF:
    def __init__(self, F, B, H, Q, R):
        self.F = F  # state transition model
        self.B = B  # control-input model
        self.H = H  # observation model
        self.Q = Q  # process noise covariance
        self.R = R  # observation noise covariance
        self.x = None  # initial state estimate
        self.P = None  # initial covariance estimate

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
