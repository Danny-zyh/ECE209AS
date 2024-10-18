import numpy as np
from scipy.interpolate import interp1d

goal = (0, 0)
def R(s_not, a, s, c=0):
    return np.array([np.array_equal(si, (0, 0)) for si in s_not]) - np.abs(a)*c

class particle:
    """
    Week 3 implementation of discretizing continuous numberline environment
    """
    def __init__(self, m=1, pc=0.3, pw=0.3, ymax=2, vmax=2, 
                    dy=0.5, dv=0.5, f_phi=lambda y: -2*np.cos(y)):
        self.m = m
        self.pc = pc
        self.pw = pw
        self.f_phi = f_phi
        self.ymax = np.round(ymax)
        self.vmax = np.round(vmax)
        self.dy, self.dv = dy, dv
        self.A = np.array([-1, 0, 1])
        self.yaxis = np.arange(-ymax, ymax+dy, dy)
        self.vaxis = np.arange(-vmax, ymax+dv, dv)
        self.yintp = interp1d(self.yaxis, self.yaxis, "nearest")
        self.vintp = interp1d(self.vaxis, self.vaxis, "nearest")

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
            abs(v)/self.vmax*self.pw/2, 
            1-abs(v)/self.vmax*self.pw, 
            abs(v)/self.vmax*self.pw/2
        ])

        crash_prob = np.array([
            1-abs(v)/self.vmax*self.pc,
            abs(v)/self.vmax*self.pc
        ])

        probs = (crash_prob[:,np.newaxis] @ wobble_prob[np.newaxis,:]).reshape(-1)
        return states, probs
    
    def fnn(self, y, v):
        '''
        Return the nearest neighbors on the grid
        Input:
            y: float, position of the particle
            v: float, velocity of the particle
        Return:
            nearest_states: ndarray, a 4 by 2 array of the four nearest states
            prob: ndarray, a 4 array with the probability of "snapping" to the nearest state
        '''
        ny1, ny2 = self.yintp(min(y+self.dy/2, self.ymax)), self.yintp(max(y-self.dy/2, -self.ymax))
        nv1, nv2 = self.vintp(min(v+self.dv/2, self.vmax)), self.vintp(max(v-self.dv/2, -self.vmax))

        nearest_states = np.array([
            [ny1, nv1], [ny1, nv2],
            [ny2, nv1], [ny2, nv2]
        ])

        # bipolar interpolation
        if ny1 == ny2:
            py1, py2 = 1, 0
        else:
            py1, py2 = 1-np.abs(y - ny1)/self.dy, 1-np.abs(y - ny2)/self.dy
        
        if nv1 == nv2:
            pv1, pv2 = 1, 0
        else:
            pv1, pv2 = 1-np.abs(v - nv1)/self.dv, 1-np.abs(v - nv2)/self.dv

        probs = np.array([py1*pv1, py1*pv2, py2*pv1, py2*pv2])

        return nearest_states, probs


    def state2id(self, y, v):
        return np.rint((y+self.ymax)/self.dy).astype(int), np.rint((v+self.vmax)/self.dv).astype(int)
    
    def v_not_plus_r(self, y, v, a, V, R, gamma):
        T_s, T_p = self.T(y, v, a)

        snap_s_p = [self.fnn(*i) for i in T_s]
        snap_p = np.array([i[-1] for i in snap_s_p])
        s = np.array([i[0] for i in snap_s_p]).reshape(-1, 2)
        p = (snap_p * T_p[:,np.newaxis]).reshape(-1)

        sid = self.state2id(s[:,0], s[:,1])
        v_not_plus_r = gamma*V[sid] + R(s, a, (y, v))
        return np.sum(v_not_plus_r * p)

    def policy_iteration(self, gamma=0.99, delta=1e-3):
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
                for y in self.yaxis:
                    for v in self.vaxis:
                        V[self.state2id(y, v)] = self.v_not_plus_r(y, v, policy[self.state2id(y, v)], V, R, gamma)
                        V[self.state2id(0, 0)] = 0  # always set the goal state to have zero value

                if np.linalg.norm(V_old - V, ord=np.inf) < delta:
                    break
        
        def policy_improvement():
            policy_stable = False
            while not policy_stable:
                policy_stable = True
                for y in self.yaxis:
                    for v in self.vaxis:
                        pi = policy[self.state2id(y, v)]
                        E_v_not_plus_r = np.array([self.v_not_plus_r(y, v, a, V, R, gamma) for a in self.A])
                        pi_new = self.A[np.argmax(E_v_not_plus_r)]
                        policy[self.state2id(y, v)] = pi_new
                        if pi != pi_new:
                            policy_stable = False

        for _ in range(100):
            policy_evaluation()
            policy_improvement()

        return V, policy

    def simulate_trajectory(self, V, policy, s0, look_ahead=True, timesteps=10):
        y, v = s0
        trajectory = [(y, v)]
        control = []

        for _ in range(timesteps):
            if look_ahead:
                fi = self.A[np.argmax([self.v_not_plus_r(y, v, a, V, R, 0.99) for a in self.A])] # do a Bellman backup at the exact position
            else:
                s_snaps, probs = self.fnn(y, v)
                s_snap = np.random.choice(s_snaps, p=probs)
                fi = policy[self.state2id(*s_snap)]

            control.append(fi)
            states, probs = self.T(y, v, fi)
            y, v = np.random.choice(states, p=probs)
            trajectory.append((y, v))




if __name__ == "__main__":
    p = particle(ymax=1, vmax=1, pc=0.3, pw=0.3, f_phi=lambda y: -np.sin(y))
    value, policy = p.policy_iteration(c=0.2)
    print("value\n", value)
    print("policy\n", policy)
