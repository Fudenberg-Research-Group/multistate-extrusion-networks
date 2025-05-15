import networkx as nx
import matplotlib.pyplot as plt

from .CohesinNetwork import CohesinNetwork


# Cyclic cohesin networks
class CyclicCohesinNetwork(CohesinNetwork):

    def __init__(self, sequence):

        self.sequence = sequence
        transitions = self.make_transitions()
        
        super().__init__(transitions)


    def make_transitions(self):
        
        transitions = [('R', self.sequence[0])]

        for s1, s2 in zip(self.sequence[:-1], self.sequence[1:]):
             transitions.append((s1, s2))
             transitions.append((s2, s1))

        transitions.append((self.sequence[-1], 'R'))

        return transitions
    

    def draw(self, color_dict,
             arc_rad=0.15,
             fig_margin=0.2,
             fig_size=(6.4,4.8),
             node_size=2000,
             edge_kwargs = {'edge_color': 'k', 'width': 1.5, 'arrowsize': 15, 'min_target_margin': 27, 'min_source_margin': 27}):
       
       positions = nx.circular_layout(self)
       positions = {node: (y, x) for (node, (x,y)) in positions.items()}
    
       curved_edges = [edge for edge in self.edges() if edge[::-1] in self.edges()]
       straight_edges = [edge for edge in self.edges() if edge[::-1] not in self.edges()]

       fig = plt.figure(figsize=fig_size)

       nx.draw_networkx_nodes(self, positions,
                              node_color=[*map(color_dict.get, self.nodes)],
                              edgecolors='k',
                              node_size=node_size)
       nx.draw_networkx_labels(self, positions)
       
       nx.draw_networkx_edges(self, positions,
                              edgelist=straight_edges,
                              **edge_kwargs)
       nx.draw_networkx_edges(self, positions,
                              edgelist=curved_edges,
                              connectionstyle=f"arc3, rad={arc_rad}",
                               **edge_kwargs)
       
       plt.margins(fig_margin)
