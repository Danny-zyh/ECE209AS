import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor

class particle:
    """
    Week 8
    """
    def __init__(self, m=1, ymax=3, vmax=3, f_phi=lambda _: 0):
        self.m = m
        self.f_phi = f_phi
        self.vmax = vmax
        self.ymax = ymax

        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0], [1 / self.m]])

        # the argmax Q is calculated by randomly sample actions
        self.sample_size = 30

    def T(self, s, fi):
        next_state = self.A @ s + self.B @ np.array([fi]) + np.array([0, self.f_phi(s)])
        return np.clip(next_state, [-self.ymax, -self.vmax], [self.ymax, self.vmax])

    def play(self, n,
                accuracy=0.1, 
                policy=lambda s: np.random.uniform(-1, 1)):
        experiences = []
        for _ in range(n):
            time_step = 0
            time_horizon = 20
            s = np.random.uniform(low=-1,high=1,size=2) * \
                    np.array([self.ymax, self.vmax])

            for time_step in range(time_horizon):
                a = policy(s)
                s_ = self.T(s, a)
                r = 1 if np.linalg.norm(s_) < accuracy else 0
                experiences.append((s, a, s_, r, time_step))
                if r == 1:
                    break
                s = s_
        return experiences

    def Q(self, s, a, nn, accuracy=0.1):
        if np.linalg.norm(s) < accuracy:
            return 0
        return nn.predict(np.array([*s, a])[np.newaxis, :])[0]
    
    def generate_dataset(self, experiences, Q, gamma=0.9):
        X = np.zeros((len(experiences), 3))
        y = np.zeros(len(experiences))
        for i, exp in enumerate(experiences):
            s, a, s, r, ts = exp
            X[i, :] = np.array([*s, a])         
            values = np.array([Q(s, a) for a in np.linspace(-1, 1, self.sample_size)])
            y[i] = (r + gamma*np.max(values))
        return X, y

    def Q_learning(self, nn, gamma=0.9):
        
        for episode in tqdm(range(100)):
            def policy(s):
                sampled_actions = np.random.uniform(low=-1, high=1, size=self.sample_size)
                value = np.array([self.Q(s, a, nn) for a in sampled_actions])
                return sampled_actions[np.argmax(value)]

            D = self.play(100, policy=policy)

            # perform Q learning on the generated experience
            X, y = self.generate_dataset(D, lambda s, a: self.Q(s, a, nn), gamma)
            nn.fit(X, y)
            
    def simulate_trajectory(self, nn, start=(0.8, 0.8), horizon=20, accuracy=0.1):
        def policy(s):
            sampled_actions = np.random.uniform(low=-1, high=1, size=self.sample_size)
            value = np.array([self.Q(s, a, nn) for a in sampled_actions])
            return sampled_actions[np.argmax(value)]

        controls = []
        trajectory = [start]
        for i in range(horizon):
            s = trajectory[-1]
            if np.linalg.norm(s) < accuracy:
                break
            a = policy(s)
            controls.append(a)
            trajectory.append(self.T(s, a))

        return trajectory, controls
