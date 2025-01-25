import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque

def compute_schedule(tasks):
    """
    1) Topologically sort the tasks.
    2) Forward pass -> ES, EF.
    3) Backward pass -> LS, LF.
    4) Float = LS - ES.
    5) Mark tasks as critical if float=0.
    Returns:
      schedule: dict {task_id: {ES, EF, LS, LF, Float, Critical}}
      project_finish: float
      topo_order: list of task IDs in topological order
    """
    # Build adjacency for topological sort
    graph = defaultdict(list)
    indeg = defaultdict(int)
    tasks_map = {}
    
    for t in tasks:
        tid = t['id']
        dur = t['duration']
        preds = t['predecessors']
        tasks_map[tid] = {"duration": dur, "predecessors": preds}
        if tid not in graph:
            graph[tid] = []
        if tid not in indeg:
            indeg[tid] = 0

    # Fill graph and indegree
    for t in tasks:
        for p in t["predecessors"]:
            graph[p].append(t["id"])
            indeg[t["id"]] += 1

    # Kahn's Algorithm for topological order
    queue = deque([n for n in indeg if indeg[n] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(topo_order) != len(tasks_map):
        raise ValueError("Graph is not a DAG (cycle detected).")

    # Forward pass: ES, EF
    ES = {}
    EF = {}
    for tid in topo_order:
        preds = tasks_map[tid]["predecessors"]
        if not preds:
            ES[tid] = 0
        else:
            ES[tid] = max(EF[p] for p in preds)
        EF[tid] = ES[tid] + tasks_map[tid]["duration"]

    project_finish = max(EF.values())

    # Build successors map for backward pass
    successors = defaultdict(list)
    for t in tasks:
        tid = t["id"]
        for p in t["predecessors"]:
            successors[p].append(tid)

    # Backward pass: LF, LS
    LF = {}
    LS = {}
    for tid in reversed(topo_order):
        succs = successors[tid]
        if not succs:
            # no successors => LF = project_finish
            LF[tid] = project_finish
        else:
            LF[tid] = min(LS[s] for s in succs)
        LS[tid] = LF[tid] - tasks_map[tid]["duration"]

    # Compute float and detect critical tasks
    schedule = {}
    for t in tasks:
        tid = t["id"]
        es = ES[tid]
        ef = EF[tid]
        ls = LS[tid]
        lf = LF[tid]
        flt = ls - es
        critical = abs(flt) < 1e-9
        schedule[tid] = {
            "ES": es,
            "EF": ef,
            "LS": ls,
            "LF": lf,
            "Float": flt,
            "Critical": critical
        }

    return schedule, project_finish, topo_order


def build_layered_positions(tasks, topo_order):
    """
    Create a left-to-right 'layered' layout based on topological levels:
      - level[node] = 1 + max(level[pred]) for each node
      - x = level[node] * horizontal_spacing
      - y = an offset depending on how many tasks share that level
    """
    tasks_map = {t["id"]: t for t in tasks}

    # 1) Compute each node's 'layer' from the topological order
    level = {}
    for node in topo_order:
        preds = tasks_map[node]["predecessors"]
        if not preds:
            level[node] = 0
        else:
            level[node] = max(level[p] for p in preds) + 1

    # 2) Group nodes by their layer
    layer_buckets = defaultdict(list)
    for node in topo_order:
        layer_buckets[level[node]].append(node)

    # 3) Assign (x, y) positions
    pos = {}
    horizontal_spacing = 3.0  # bigger => more spread horizontally
    vertical_spacing = 3.0    # bigger => more spread vertically

    for lvl, nodes in layer_buckets.items():
        # Center the nodes in this level
        # Let's distribute them along the y-axis
        for idx, node in enumerate(nodes):
            x = lvl * horizontal_spacing
            # negative y so that topmost node has the smallest idx
            y = -idx * vertical_spacing
            pos[node] = (x, y)

    return pos


def find_critical_path(tasks, schedule):
    """
    Build a subgraph of critical tasks (where EF(u)=ES(v) for consecutive tasks).
    Return one critical path from a 'start' (no preds) to an 'end' (no succs).
    """
    from collections import defaultdict

    # Build adjacency for original graph
    successors_map = defaultdict(list)
    preds_map = defaultdict(list)
    for t in tasks:
        tid = t["id"]
        for p in t["predecessors"]:
            successors_map[p].append(tid)
            preds_map[tid].append(p)

    # Identify critical tasks
    is_crit = {tid: schedule[tid]["Critical"] for tid in schedule}
    EF_map = {tid: schedule[tid]["EF"] for tid in schedule}
    ES_map = {tid: schedule[tid]["ES"] for tid in schedule}

    # Build adjacency for *critical* edges only
    crit_graph = defaultdict(list)
    for u in schedule:
        for v in successors_map[u]:
            if is_crit[u] and is_crit[v]:
                # Edge is critical if EF(u) == ES(v)
                if abs(EF_map[u] - ES_map[v]) < 1e-9:
                    crit_graph[u].append(v)

    # Start tasks = no preds
    start_tasks = [t["id"] for t in tasks if not t["predecessors"] and is_crit[t["id"]]]
    # End tasks = tasks that have no critical successors
    end_tasks = []
    for tid in schedule:
        if tid not in crit_graph or len(crit_graph[tid]) == 0:
            if is_crit[tid]:
                end_tasks.append(tid)

    # DFS to find one path : depth first search
    visited = set()
    path = []

    def dfs(curr):
        visited.add(curr)
        path.append(curr)
        if curr in end_tasks:
            return True
        for nxt in crit_graph[curr]:
            if nxt not in visited:
                if dfs(nxt):
                    return True
        path.pop()
        return False

    for s in start_tasks:
        if dfs(s):
            return path
    return []
def plot_mpm(tasks, schedule, critical_path, pos):
    import networkx as nx
    import matplotlib.pyplot as plt
    from collections import defaultdict

    G = nx.DiGraph()
    # Add nodes
    for tid, info in schedule.items():
        G.add_node(tid, **info)

    # Build adjacency from tasks
    adj_map = defaultdict(list)
    for t in tasks:
        for p in t["predecessors"]:
            adj_map[p].append(t["id"])
    # Add edges
    for u in adj_map:
        for v in adj_map[u]:
            G.add_edge(u, v)

    # Determine edges on the critical path:
    crit_edges = set()
    for i in range(len(critical_path) - 1):
        crit_edges.add((critical_path[i], critical_path[i+1]))

    # Node colors and labels
    node_colors = []
    node_labels = {}
    for node, data in G.nodes(data=True):
        if data["Critical"]:
            node_colors.append("red")  # critical tasks
        else:
            node_colors.append("lightblue")

        # multi-line label
        lbl = (
            f"{node}\n"
            f"ES={data['ES']} EF={data['EF']}\n"
            f"LS={data['LS']} LF={data['LF']}"
        )
        node_labels[node] = lbl

    # Edge colors
    edge_colors = []
    for (u, v) in G.edges():
        if (u, v) in crit_edges:
            edge_colors.append("red")
        else:
            edge_colors.append("black")

    plt.figure(figsize=(10, 6))

    # --- KEY PART: Force arrow styles and bigger size ---
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        node_size=3000,                # Must match the node_size used below
        arrowstyle="-|>",             # or "->" or another style
        arrowsize=20,                 # bigger arrowheads
        connectionstyle="arc3,rad=0.05"  # a slight curve can help
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, 
        pos,
        node_color=node_colors, 
        node_size=3000, 
        node_shape='s', 
        edgecolors='black'
    )
    # Draw labels
    nx.draw_networkx_labels(
        G, 
        pos,
        labels=node_labels, 
        font_size=9, 
        font_color='black'
    )

    plt.title("Layered MPM Diagram - With Arrowheads")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example tasks
    tasks_data = [
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 2, "predecessors": ["A"]},
        {"id": "C", "duration": 4, "predecessors": ["A"]},
        {"id": "D", "duration": 2, "predecessors": ["B", "C"]},
        {"id": "E", "duration": 1, "predecessors": ["D"]},
    ]

    # 1) Compute the schedule + topological order
    schedule, finish_time, topo_order = compute_schedule(tasks_data)

    print("Schedule Results:")
    print("Task | ES | EF | LS | LF | Float | Critical")
    for tid in sorted(schedule.keys()):
        s = schedule[tid]
        print(f"{tid:4} | {s['ES']:2} | {s['EF']:2} | {s['LS']:2} | {s['LF']:2} | {s['Float']:5.1f} | {s['Critical']}")

    print(f"\nProject Finish Time: {finish_time:.1f}")

    # 2) Identify one critical path
    cp = find_critical_path(tasks_data, schedule)
    if cp:
        print("One Critical Path:", " → ".join(cp))
    else:
        print("No single critical path found (maybe multiple).")

    # 3) Build a left-to-right layered layout
    pos = build_layered_positions(tasks_data, topo_order)

    # 4) Plot the MPM diagram
    plot_mpm(tasks_data, schedule, cp, pos)
 