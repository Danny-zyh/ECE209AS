import numpy as np
import control as ct

class particle:
    """
    Week 5
    """
    def __init__(self, m=1, ymax=2, vmax=2):
        self.m = m
        self.ymax = ymax
        self.vmax = vmax

    def T(self, y, v, fi, nl=0.1):
        '''
        Calculate the transition probability
        Input:
            y: float, position
            v: float, velocity
            fi: float, input force
            nl: float, noise level (standard deviation of gaussian noise)
        Output:
            states: array, an array of possible state
            probs: array, an array of probabilities of arriving at the states
        '''
        y_next = y + v
        v_next = v + (1 / self.m) * fi + np.random.normal(0, scale=nl)

        y_next = np.clip(y_next, -self.ymax, self.ymax)
        v_next = np.clip(v_next, -self.vmax, self.vmax)

        return np.array([y_next, v_next])


    def lqr(self, Q=np.diag(np.ones(2)), R=1):
        # System dynamics matrices
        A = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1 / self.m]])

        # Calculate LQR gain using control.lqr
        K, S, E = ct.dlqr(A, B, Q, R)
        return K, S, E

    def simulate_trajectory(self, start, Q=np.diag(np.ones(2)), R=10, timestep=50):
        state = np.array(start)
        K, *_ = self.lqr(Q, R)
        actions = []
        trajectory = [tuple(state)]
        t = 0
        while np.linalg.norm(state - np.zeros(2)) > 1e-2:
            if t > timestep:
                break
            t += 1
            action = -K @ state
            actions.append(tuple(action))
            state = np.array(self.T(*tuple(state), action[0]))
            trajectory.append(tuple(state))
        return trajectory, actions