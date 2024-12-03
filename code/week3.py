import numpy as np
from scipy.interpolate import RegularGridInterpolator

# def R(s_next, a, s, c=0, goal=(0, 0)):
#     return np.array([np.array_equal(si, goal) for si in s_next]) - np.abs(a)*c

class particle:
    """
    Week 3 implementation of discretizing continuous numberline environment
    """
    def __init__(self, m=1, pc=0.3, pw=0.3, ymax=2, vmax=2, 
                    dy=0.5, dv=0.5, A=[-1, 0, 1], f_phi=lambda y: -2*np.cos(y)):
        self.m = m
        self.A = A
        self.pc = pc
        self.pw = pw
        self.f_phi = f_phi
        self.ymax = np.rint(ymax)
        self.vmax = np.rint(vmax)
        self.dy, self.dv = dy, dv
        self.yaxis = np.arange(-ymax, ymax+dy, dy)
        self.vaxis = np.arange(-vmax, ymax+dv, dv)

    def T(self, y, v, fi):
        '''
        Calculate the transition states as well as their probability
        Input:
            y: float, position
            v: float, velocity
            fi: float, input force
        Output:
            states: array, an array of possible state
            probs: array, an array of probabilities of arriving at the states
        '''
        y += v
        v += (1 / self.m) * (fi + self.f_phi(y))

        y = np.clip(y, -self.ymax, self.ymax)
        v = np.clip(v, -self.vmax, self.vmax)

        states = np.array([
            [y, max(v-1, -self.vmax)], 
            [y, v], 
            [y, min(v+1, self.vmax)], 
            [y, 0]  , [y, 0], [y, 0]])

        wobble_prob = np.array([
            np.abs(v)/self.vmax*self.pw/2, 
            1-np.abs(v)/self.vmax*self.pw, 
            np.abs(v)/self.vmax*self.pw/2
        ])

        crash_prob = np.array([
            1-np.abs(v)/self.vmax*self.pc,
            np.abs(v)/self.vmax*self.pc
        ])

        probs = (crash_prob[:,np.newaxis] @ wobble_prob[np.newaxis,:]).reshape(-1)
        return states, probs

    def state2id(self, y, v):
        return np.rint((y+self.ymax)/self.dy).astype(int), np.rint((v+self.vmax)/self.dv).astype(int)
    
    def E_discounted_reward(self, y, v, a, V_interp, R, gamma):
        states, prob = self.T(y, v, a)
        discounted_reward = gamma*V_interp(states) + R(states, a, (y, v))
        return np.sum(discounted_reward * prob)

    def policy_iteration(self, R, gamma=0.99, delta=1e-3, goal=(0, 0)):
        '''
        Value iteration to find the optimal policy
        Input:
            goal: goal state of the system in which you will receive reward 1
            gamma: time discount faster
            c: fuel consumption for each non-zero fi
            delta: stopping criteria
        Output:
            V: array, value function given state
            policy: array, optimal policy given state
        '''

        state_shape = (len(self.yaxis), len(self.vaxis))
        V = np.zeros(state_shape)
        policy = np.zeros(state_shape)

        def policy_evaluation():
            for _ in range(1000):
                V_old = V.copy()
                V_interp = RegularGridInterpolator((self.yaxis, self.vaxis), V)
                for y in self.yaxis:
                    for v in self.vaxis:
                        a = policy[self.state2id(y, v)]
                        V[self.state2id(y, v)] = self.E_discounted_reward(y, v, a, V_interp, R, gamma)
                        V[self.state2id(*goal)] = 0  # always set the goal state to have zero value

                if np.linalg.norm(V_old - V, ord=np.inf) < delta:
                    break
        
        def policy_improvement():
            policy_stable = False
            V_interp = RegularGridInterpolator((self.yaxis, self.vaxis), V)
            while not policy_stable:
                policy_stable = True
                for y in self.yaxis:
                    for v in self.vaxis:
                        pi = policy[self.state2id(y, v)]
                        Q_a = [
                            self.E_discounted_reward(y, v, a, V_interp, R, gamma) 
                            for a in self.A
                        ]
                        pi_new = self.A[np.argmax(Q_a)]
                        policy[self.state2id(y, v)] = pi_new
                        if pi != pi_new:
                            policy_stable = False

        for _ in range(100):
            policy_evaluation()
            policy_improvement()

        return V, policy

    def simulate_trajectory(self, V, policy, y0, v0, R, 
                                gamma=0.99, 
                                goal=(0,0), 
                                look_ahead=True, 
                                mode="4nn", 
                                timesteps=100):
        y, v = y0, v0
        V_interp = RegularGridInterpolator((self.yaxis, self.vaxis), V)
        trajectory = [(y, v)]
        control = []

        t = 0
        while not np.allclose((y, v), goal, atol=self.dv):
            if t > timesteps:
                break
            t += 1
            if look_ahead:
                # if look_ahead we assume only use 4nn
                Q_a = [
                    self.E_discounted_reward(y, v, a, V_interp, R, gamma) 
                    for a in self.A
                ]
                fi = self.A[np.argmax(Q_a)]
            else:
                if mode == "1nn":
                    y_snap, v_snap = np.rint(y/self.dy)*self.dy, np.rint(v/self.dv)*self.dv
                elif mode == "4nn":
                    yh, yl = np.ceil(y/self.dy)*self.dy, np.floor(y/self.dy)*self.dy
                    vh, vl = np.ceil(v/self.dv)*self.dv, np.floor(v/self.dv)*self.dv

                    py, pv = (y - yl) / self.dy, (v - vl) / self.dv
                    y_snap = yl if np.random.uniform() < py else yh
                    v_snap = vl if np.random.uniform() < pv else vh

                fi = policy[self.state2id(y_snap, v_snap)]

            control.append(fi)
            states, probs = self.T(y, v, fi)
            idx = np.random.choice(len(states), p=probs)
            y, v = states[idx]
            trajectory.append((y, v))

        return np.array(trajectory), np.array(control)


if __name__ == "__main__":
    p = particle(ymax=1, vmax=1, pc=0.3, pw=0.3, f_phi=lambda y: -np.sin(y))
    value, policy = p.policy_iteration(c=0.2)
    print("value\n", value)
    print("policy\n", policy)
