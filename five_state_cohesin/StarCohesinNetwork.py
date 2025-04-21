import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .CohesinNetwork import CohesinNetwork


# Star cohesin networks
class StarCohesinNetwork(CohesinNetwork):

    def __init__(self, sequence):

        self.sequence = sequence
        transitions = self.make_transitions()
        
        super().__init__(transitions)


    def make_transitions(self, root_id=0):
        
        transitions = []
        full_sequence = list(self.sequence) + ['R']

        for i, s in enumerate(full_sequence):
             if i != root_id:
                transitions.append((self.sequence[root_id], s))
                transitions.append((s, self.sequence[root_id]))

        return transitions
    

    def draw(self, color_dict,
             arc_rad=0.15,
             fig_margin=0.2,
             fig_size=(4.8,4.8),
             node_size=2000,
             edge_kwargs = {'edge_color': 'k', 'width': 1.5, 'arrowsize': 15, 'min_target_margin': 27, 'min_source_margin': 27}):
       
       N = self.number_of_nodes()
       
       positions_array = np.zeros((N, 2))
       theta = np.linspace(-np.pi, np.pi, num=N-1, endpoint=False)
	       
       positions_array[1:, 0] = np.cos(theta)
       positions_array[1:, 1] = np.sin(theta)

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
