import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def bfs(graph, start, visited):
    """Perform BFS breadth first search from 'start', marking all reachable nodes, and return the list of visited in this BFS."""
    queue = deque([start])
    visited[start] = True
    component = [start]

    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                component.append(neighbor)
    return component

def find_connected_components(graph):
    """Return a list of connected components. Each component is a list of node IDs."""
    all_vertices = sorted(graph.keys())
    visited = {v: False for v in all_vertices}
    components = []

    for v in all_vertices:
        if not visited[v]:
            comp = bfs(graph, v, visited)
            components.append(comp)

    return components

def plot_undirected_graph(graph):
    """
    Build a NetworkX graph from 'graph' (an adjacency list), 
    find connected components, then plot, coloring each component differently.
    Finally, indicate if the graph is connected or not.
    """
    G = nx.Graph()
    
    # 1) Add edges (undirected) to NetworkX
    for u in graph:
        for v in graph[u]:
            G.add_edge(u, v)
    
    # 2) Find connected components
    components = find_connected_components(graph)
    num_components = len(components)

    # 3) Is the graph connected?
    all_nodes = list(graph.keys())
    total_nodes = len(all_nodes)
    # "Connected" if exactly one component that includes ALL nodes
    is_connected = (num_components == 1 and len(components[0]) == total_nodes)
    if is_connected:
        print("Le graphe est CONNEXE.")
    else:
        print("Le graphe N'EST PAS connexe.")
        print(f"Il a {num_components} composantes connexes:")
        for i, comp in enumerate(components, start=1):
            print(f"  Composante {i}: {comp}")

    # 4) Color each connected component differently
    palette = ["red", "green", "blue", "orange", "cyan", "magenta", "yellow"]
    node_color_map = {}
    comp_index = 0
    for comp in components:
        color = palette[comp_index % len(palette)]
        for node in comp:
            node_color_map[node] = color
        comp_index += 1
    
    # 5) Prepare color list in node order
    sorted_nodes = sorted(G.nodes())
    colors = [node_color_map[n] for n in sorted_nodes]
    
    # 6) Layout
    pos = nx.spring_layout(G, seed=42)

    # 7) Plot
    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800)
    nx.draw_networkx_labels(G, pos, labels={n:str(n) for n in sorted_nodes}, font_color="white")
    nx.draw_networkx_edges(G, pos, width=2)

    # 8) Title
    if is_connected:
        plt.title("Graph Connexe (1 composante).")
    else:
        plt.title(f"Graph NON Connexe ({num_components} composantes).")

    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example graph with edges (1,2), (2,3), (3,4), (5,6), (6,7)
    # => Two components: {1,2,3,4} and {5,6,7}
    graph_example = {
        1: [2],
        2: [1, 3],
        3: [2, 4],
        4: [3],
        5: [6],
        6: [5, 7],
        7: [6]
    }

    plot_undirected_graph(graph_example)
