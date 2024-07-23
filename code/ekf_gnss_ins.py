import numpy as np

class EKF:
    def __init__(self, f, h, F, H, Q, R, P, x0):
        self.f = f  # State transition function
        self.h = h  # Observation function
        self.F = F  # Jacobian of the state transition function
        self.H = H  # Jacobian of the observation function
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial estimate error covariance
        self.x = x0 # Initial state estimate

    def predict(self, u):
        # Predict the state and estimate covariance
        self.x = self.f(self.x, u)
        self.P = self.F(self.x, u) @ self.P @ self.F(self.x, u).T + self.Q

    def update(self, z):
        # Compute the Kalman gain
        S = self.H(self.x) @ self.P @ self.H(self.x).T + self.R
        K = self.P @ self.H(self.x).T @ np.linalg.inv(S)

        # Update the state estimate and error covariance
        y = z - self.h(self.x)
        self.x = self.x + K @ y
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H(self.x)) @ self.P

# Example usage
def f(x, u):
    # State transition function
    dt = 1.0  # Time step
    F = np.array([[1, dt], [0, 1]])
    return F @ x + np.array([0, u])

def h(x):
    # Observation function
    return np.array([x[0]])

def F_jacobian(x, u):
    # Jacobian of the state transition function
    dt = 1.0
    return np.array([[1, dt], [0, 1]])

def H_jacobian(x):
    # Jacobian of the observation function
    return np.array([[1, 0]])

# Initial state
x0 = np.array([0, 1])

# Initial estimate covariance
P = np.array([[1000, 0], [0, 1000]])

# Process noise covariance
Q = np.array([[1, 0], [0, 3]])

# Measurement noise covariance
R = np.array([[10]])

# Create EKF instance
ekf = EKF(f, h, F_jacobian, H_jacobian, Q, R, P, x0)

# Measurements and control inputs
measurements = [1, 2, 3, 4, 5, 6]
controls = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Run EKF
for z, u in zip(measurements, controls):
    ekf.predict(u)
    ekf.update(np.array([z]))
    print(f"Post-update state: {ekf.x}")
