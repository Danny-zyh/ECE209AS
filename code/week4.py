import numpy as np

class particle:
    """
    Week 4
    """
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

    def straightline_motion(self, state1, state2, max_duration=10):
        """
        Check if the particle can move from state1 to state2 using a force -1 <= f <= 1.
        Parameters:
            state1: tuple, initial state (y1, v1)
            state2: tuple, target state (y2, v2)
            max_duration: maximum time allowed for the transition.
        Output:
            transitions: list of tuples, each containing the force applied and the time taken for the transition.
        """
        y1, v1 = state1
        y2, v2 = state2
        transitions = []

        for t in np.arange(1 , max_duration):  # Iterate over continuous time steps
            # Solve for the required force using the final velocity equation
            required_force = self.m * (v2 - v1) / t

            # Check if the required force is within the allowable bounds
            if -1 <= required_force <= 1:
                # Now, check if the required force brings the particle to the target position
                predicted_position = y1 + v1 * t + 0.5 * (required_force / self.m) * t**2

                if y2 == predicted_position:  # Check if position matches with tolerance
                    # Record the valid transition
                    transitions.append((required_force, t))

        return transitions

if __name__ == "__main__":
    # TODO
    exit()