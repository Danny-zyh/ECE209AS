import numpy as np
import networkx as nx
from scipy.spatial import KDTree

class particle:
    """
    Week 4
    """
    def __init__(self, m=1, ymax=1, vmax=1, f_phi=lambda y: 0):
        self.m = m
        self.f_phi = f_phi
        self.ymax = ymax
        self.vmax = vmax

    def T(self, y, v, fi, noise_amplitude=0):
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
        y_next = y + v
        v_next = v + (1 / self.m) * (fi + self.f_phi(y))

        v_next += np.random.normal(loc=0, scale=noise_amplitude)

        y_next = np.clip(y_next, -self.ymax, self.ymax)
        v_next = np.clip(v_next, -self.vmax, self.vmax)

        return y_next, v_next
    
    def two_step_dynamic(self, y, v, f1, f2):
        y2 = (y+2*v) + 1/self.m * (self.f_phi(y) + f1)
        v2 = v + 1/self.m * (f1 + f2 + self.f_phi(y) + self.f_phi(y+v))

        return y2, v2
    
    def inverse_2_step_dynamic(self, s1, s2):
        y, v = s1
        y_naught, v_naught = s2
        f1 = (y_naught - (y+2*v) - 1/self.m*self.f_phi(y))*self.m
        f2 = (v_naught - v - 1/self.m*(f1+self.f_phi(y)+self.f_phi(y+v)))*self.m
        return f1, f2

    def connectable(self, s1, s2):
        """
        Check if the particle can move from state1 to state2 in two steps
        Parameters:
            state1: tuple, initial state (y1, v1)
            state2: tuple, target state (y2, v2)
        """
        f1, f2 = self.inverse_2_step_dynamic(s1, s2)
        return -1 <= f1 <= 1 and -1 <= f2 <= 1

    def build_prm(self, G=None, N=100, radius=5):
        if G is None:
            G = nx.DiGraph()

        ys = np.random.uniform(low=-self.ymax, high=self.ymax, size=N)
        vs = np.random.uniform(low=-self.vmax, high=self.vmax, size=N)

        data = np.array([ys, vs]).T

        kdt = KDTree(data)

        G.add_nodes_from([(y, v) for y, v in zip(ys, vs)])

        for node in G.nodes:
            neighbors = data[kdt.query_ball_point(node, radius), :]
            for neighbor in neighbors:
                if self.connectable(neighbor, node):
                    if not nx.algorithms.has_path(G, tuple(neighbor), node):
                        G.add_edge(tuple(neighbor), node)
                if self.connectable(node, neighbor):
                    if not nx.algorithms.has_path(G, tuple(neighbor), node):
                        G.add_edge(node, tuple(neighbor))

        return G

    def find_path(self, G, s1, s2, noise_amplitude=0):
        starts = filter(lambda n: self.connectable(s1, n), G.nodes)
        ends = filter(lambda n: self.connectable(n, s2), G.nodes)

        for s in starts:
            for e in ends:
                if nx.algorithms.has_path(G, s, e):
                    path = [s1] + nx.shortest_path(G, s, e) + [s2]
                    force = []
                    for i in range(len(path)-1):
                        force += list(self.inverse_2_step_dynamic(path[i], path[i+1]))

                    trajectory = [s1]
                    for fi in force:
                        trajectory.append(self.T(*trajectory[-1], fi, noise_amplitude))
                    return trajectory, force
        
        return None