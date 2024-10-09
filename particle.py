import numpy as np

class particle:
    def __init__(self, m=1, pc=0.3, pw=0.3, ymax=2, vmax=2, f_phi=lambda y: -2*np.cos(y)):
        self.m = m
        self.pc = pc
        self.pw = pw
        self.f_phi = f_phi
        self.ymax = ymax
        self.vmax = vmax

    def T(self, y, v, fi):
        '''
        Calculate the transition probability
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

        probs = (crash_prob[:,np.newaxis] @ wobble_prob[np.newaxis,:]).flatten()
        return states, probs

    def value_iteration(self, goal=(0,0), gamma=0.5, c=0.2, delta=1e-3):
        '''
        Value iteration to find the optimal policy
        Input:
            goal: goal state of the system in which you will receive reward 1
            gamma: time discount fater
            c: fuel consumption for each non-zero fi
            delta: stopping criteria
        Output:
            V: array, value function given state
            policy: array, optimal policy given state
        '''
        def state2id(y, v):
            return np.rint(y+self.ymax).astype(int), np.rint(v+self.vmax).astype(int)
        
        def R(s_not, a, s):
            return np.array([np.array_equal(si, goal) for si in s_not]) - np.abs(a)*c

        state_shape = (2*self.ymax+1, 2*self.vmax+1)
        V = np.zeros(state_shape)
        policy = np.zeros(state_shape)

        A = np.array([-1, 0, 1])  # action space
        for _ in range(1000):
            V_old = V.copy()
            for y in range(-self.ymax, self.ymax+1):
                for v in range(-self.vmax, self.vmax+1):

                    def v_not_plus_r(a):
                        s, p = self.T(y, v, a)
                        sid = state2id(s[:,0], s[:,1])
                        v_not_plus_r = gamma*V[sid] + R(s, a, (y, v))
                        return np.sum(v_not_plus_r * p)

                    E_v_not_plus_r = np.array([v_not_plus_r(a) for a in A])

                    V[state2id(y, v)] = np.max(E_v_not_plus_r)
                    policy[state2id(y, v)] = A[np.argmax(E_v_not_plus_r)]
                    V[state2id(goal[0], goal[1])] = 0  # always set the goal state to have zero value

            # early stopping
            if np.linalg.norm(V_old - V, ord=np.inf) < delta:
                break
        return V, policy


if __name__ == "__main__":
    p = particle(ymax=1, vmax=1, pc=0, pw=0, f_phi=lambda y: 0)
    value, policy = p.value_iteration()
    print("value\n", value)
    print("policy\n", policy)
