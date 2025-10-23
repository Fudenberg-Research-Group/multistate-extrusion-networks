import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .CohesinNetwork import CohesinNetwork


# Linear cohesin networks
class LinearCohesinNetwork(CohesinNetwork):

    def __init__(self, sequence):

        self.sequence = sequence
        transitions = self.make_transitions()
        
        super().__init__(transitions)


    def make_transitions(self):
        
        transitions = []
        full_sequence = ['R'] + list(self.sequence)

        for s1, s2 in zip(full_sequence[:-1], full_sequence[1:]):
             transitions.append((s1, s2))
             transitions.append((s2, s1))

        return transitions
    

    def draw(self, color_dict,
             arc_rad=0.15,
             fig_margin=0.2,
             fig_size=(10,4.8),
             node_size=2000,
             edge_kwargs = {'edge_color': 'k', 'width': 1.5, 'arrowsize': 15, 'min_target_margin': 27, 'min_source_margin': 27}):
       
       N = self.number_of_nodes()
       
       positions_array = np.vstack((np.arange(N)-(N-1)/2., np.zeros(N))).T
       positions = {node: tuple(p) for (node, p) in zip(self.nodes, positions_array)}
    
       fig = plt.figure(figsize=fig_size)

       nx.draw_networkx_nodes(self, positions,
                              node_color=[*map(color_dict.get, self.nodes)],
                              edgecolors='k',
                              node_size=node_size)
       nx.draw_networkx_labels(self, positions)
       
       nx.draw_networkx_edges(self, positions,
                              connectionstyle=f"arc3, rad={arc_rad}",
                               **edge_kwargs)
       
       plt.margins(fig_margin)
