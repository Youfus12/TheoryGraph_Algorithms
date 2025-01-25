import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class BellmanFordVisualizer:
    def __init__(self, graph, source):
        self.graph = graph
        self.source = source
        self.G = nx.DiGraph()
        self.initialize_graph()
        self.pos = self.hierarchical_layout()
        self.colors = plt.cm.viridis
        self.edge_highlight = '#ff4757'
        self.node_highlight = '#2ed573'
        
    def initialize_graph(self):
        for u in self.graph:
            for v, w in self.graph[u]:
                self.G.add_edge(u, v, weight=w, color='#57606f', width=1)
                
    def hierarchical_layout(self):
        pos = {
            1: np.array([0, 4]),
            2: np.array([-2, 2]),
            3: np.array([2, 2]),
            4: np.array([-1, 0]),
            5: np.array([1, 0])
        }
        return pos
    
    def run_algorithm(self):
        self.distances = {n: float('inf') for n in self.G.nodes}
        self.predecessors = {n: None for n in self.G.nodes}
        self.distances[self.source] = 0
        
        for _ in range(len(self.G.nodes)-1):
            for u, v, data in self.G.edges(data=True):
                if self.distances[u] + data['weight'] < self.distances[v]:
                    self.distances[v] = self.distances[u] + data['weight']
                    self.predecessors[v] = u

        self.negative_cycle = False
        for u, v, data in self.G.edges(data=True):
            if self.distances[u] + data['weight'] < self.distances[v]:
                self.negative_cycle = True
                break
                
    def final_visualization(self):
        self.run_algorithm()
        
        # Create single figure instance
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        ax.set_facecolor('white')
        
        # Update edge styles for optimal paths
        for v in self.G.nodes:
            u = self.predecessors.get(v)
            if u is not None:
                self.G.edges[u, v]['color'] = self.edge_highlight
                self.G.edges[u, v]['width'] = 3.5

        # Draw edges
        edge_colors = [data['color'] for _, _, data in self.G.edges(data=True)]
        edge_widths = [data['width'] for _, _, data in self.G.edges(data=True)]
        
        nx.draw_networkx_edges(
            self.G, self.pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=25,
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Edge labels
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.G, self.pos,
            edge_labels=edge_labels,
            ax=ax,
            font_color='black',
            font_size=12,
            label_pos=0.75
        )
        
        # Nodes with increased size
        node_colors = [
            self.node_highlight if n == self.source 
            else '#a4b0be' for n in self.G.nodes
        ]
        
        nodes = nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color=node_colors,
            node_size=3500,
            alpha=0.95,
            ax=ax
        )
        nodes.set_edgecolor('black')
        
        # Labels with black text
        labels = {
            n: f"Node {n}\nDistance: {self.distances[n] if self.distances[n] != float('inf') else '∞'}"
            for n in self.G.nodes
        }
        
        nx.draw_networkx_labels(
            self.G, self.pos,
            labels=labels,
            ax=ax,
            font_size=12,
            font_color='black',
            font_weight='bold'
        )
        
        # Title
        title = f"Final Result - Shortest Paths from Node {self.source}"
        if self.negative_cycle:
            title += "\n⚠ Negative Cycle Detected! ⚠"
            
        ax.set_title(title, color='black', fontsize=16, pad=20)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.edge_highlight, lw=3, label='Optimal Path'),
            plt.Line2D([0], [0], color='#57606f', lw=2, label='Regular Edge'),
            plt.Line2D([0], [0], marker='o', color='w', markersize=15,
                      markerfacecolor=self.node_highlight, label='Source Node')
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            facecolor='#f0f0f0',
            edgecolor='none',
            labelcolor='black'
        )
        
        ax.grid(False)
        plt.tight_layout()
        plt.show()

# Usage
graph = {
    1: [(2, 6), (3, 7)],
    2: [(4, 5), (3, 8)],
    3: [(4, -3)],
    4: [(5, -4)],
    5: [(2, 2)]
}

visualizer = BellmanFordVisualizer(graph, 1)
visualizer.final_visualization()