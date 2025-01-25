import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.patches import Patch

def kosaraju_scc(graph):
    """Kosaraju's algorithm to find strongly connected components"""
    visited = {node: False for node in graph}
    order = []
    
    # First DFS pass to get finishing order
    def dfs_finish(node):
        stack = [(node, False)]
        while stack:
            current, processed = stack.pop()
            if processed:
                order.append(current)
                continue
            if visited[current]:
                continue
            visited[current] = True
            stack.append((current, True))
            for neighbor in reversed(graph.get(current, [])):
                if not visited[neighbor]:
                    stack.append((neighbor, False))
    
    for node in graph:
        if not visited[node]:
            dfs_finish(node)
    
    # Create transposed graph
    transposed = {node: [] for node in graph}
    for u in graph:
        for v in graph[u]:
            transposed[v].append(u)
    
    # Second DFS pass on transposed graph
    visited = {node: False for node in graph}
    scc = []
    
    for node in reversed(order):
        if not visited[node]:
            component = []
            stack = [node]
            visited[node] = True
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in transposed[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            scc.append(sorted(component))
    
    return scc

def analyze_and_plot(graph):
    """Analyze and visualize the graph with SCCs"""
    G = nx.DiGraph()
    
    # Build the graph
    for u in graph:
        for v in graph[u]:
            G.add_edge(u, v)
    
    # Find strongly connected components
    components = kosaraju_scc(graph)
    
    # Print results
    print("Composantes fortement connexes:")
    for i, comp in enumerate(components, 1):
        print(f"Composante {i}: {comp}")
    
    # Color mapping
    colors = plt.cm.tab10.colors
    node_colors = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_colors[node] = colors[idx % len(colors)]
    
    # Create layout and plot
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    
    # Draw nodes with component colors
    nx.draw_networkx_nodes(G, pos, node_color=[node_colors[n] for n in G.nodes()], 
                          node_size=800, alpha=0.9)
    
    # Draw labels and edges
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, 
                          edge_color='gray', width=1.5)
    
    # Create legend with corrected facecolor parameter
    legend_elements = [Patch(facecolor=colors[i], 
                            label=f'Composante {i+1}: {comp}')
                      for i, comp in enumerate(components)]
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Composantes Fortement Connexes", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Test with the specific example
example_graph = {
    1: [2],
    2: [3],
    3: [1],    # First cycle: 1-2-3
    4: [5],
    5: [6],
    6: [4]     # Second cycle: 4-5-6
}

# Run the analysis and visualization
analyze_and_plot(example_graph)