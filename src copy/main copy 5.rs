type Node = u64;
type Weight = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Edge {
    from: Node,
    to: Node,
    weight: Weight,
}

struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

fn print_graph(graph: &Graph) {
    for edge in &graph.edges {
        println!("{} --{}--> {}", edge.from, edge.weight, edge.to);
    }
}
struct Path {
    from: u64,
    to: u64,
    steps: Vec<(Node, Weight, Node)>,
}

fn print_path(path: &Path) {
    if path.steps.is_empty() {
        println!("{} --{}-> {}", path.from, '∞', path.to);
        return;
    }
    let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
    print!("{} --{}-> {} : ", path.from, sum, path.to);
    for (from, weight, to) in &path.steps {
        print!("{}--{}->{}|", from, weight, to);
    }
    println!();
}

// Compute the shortest paths from every node to every other node
fn dijkstra(graph: &Graph, start: Node, end: Node, max_time: u64) -> Path {
    let mut dist = vec![u32::MAX; graph.edges.len()];
    let mut prev = vec![u64::MAX; graph.edges.len()];
    let mut visited = vec![false; graph.edges.len()];
    dist[start as usize] = 0;
    for _ in 0..graph.edges.len() {
        let mut min_dist = u32::MAX;
        let mut min_node = u64::MAX;
        for (i, d) in dist.iter().enumerate() {
            if !visited[i] && *d < min_dist {
                min_dist = *d;
                min_node = i as u64;
            }
        }
        if min_node > max_time {
            break;
        }
        visited[min_node as usize] = true;
        for edge in &graph.edges {
            if edge.from == min_node {
                if dist[min_node as usize] == u32::MAX || edge.weight == u32::MAX {
                    continue;
                }
                let alt = dist[min_node as usize] + edge.weight;
                if alt < dist[edge.to as usize] {
                    dist[edge.to as usize] = alt;
                    prev[edge.to as usize] = min_node;
                }
            }
        }
    }
    let mut steps = vec![];
    let mut node = end;
    while node != start {
        let prev_node = prev[node as usize];
        if prev_node == u64::MAX {
            break;
        }
        let weight = dist[node as usize] - dist[prev_node as usize];
        steps.push((prev_node, weight, node));
        node = prev_node;
    }
    steps.reverse();
    return Path {
        from: start,
        to: end,
        steps,
    };
}

fn johnson(graph: &Graph, max_time: u64) -> Vec<Path> {
    let mut paths = vec![];
    // find all the nodes in the graph
    let mut nodes = vec![];
    for edge in &graph.edges {
        if !nodes.contains(&edge.from) {
            nodes.push(edge.from);
        }
        if !nodes.contains(&edge.to) {
            nodes.push(edge.to);
        }
    }
    for start in &nodes {
        for end in &nodes {
            if start != end {
                let path = dijkstra(graph, *start, *end, max_time);
                paths.push(path);
            }
        }
    }
    return paths;
}


type DistanceMatrix = Vec<Vec<u32>>;

fn from_shortest_paths(paths: &Vec<Path>, nodes: &Vec<Node>) -> DistanceMatrix {
    let mut matrix = vec![vec![u32::MAX; nodes.len()]; nodes.len()];
    // fill the diagonal with 0
    for i in 0..nodes.len() {
        matrix[i][i] = 0;
    }
    for path in paths {
        if path.steps.is_empty() {
            continue;
        }
        let from = nodes.iter().position(|&n| n == path.from).unwrap();
        let to = nodes.iter().position(|&n| n == path.to).unwrap();
        let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
        matrix[from][to] = sum;
    }
    return matrix;
}

fn print_matrix(matrix: &DistanceMatrix) {
    for row in matrix {
        for cell in row {
            let cell_to_str = if *cell == u32::MAX {
                "∞".to_string()
            } else {
                cell.to_string()
            };
            print!("{}, ", cell_to_str);
        }
        println!();
    }
}

struct TimeVaryingGraph {
    max_time: u64,
    edges: Vec<Edge>,
    deleted_edges: Vec<(u64, u64, u64)>,
}

struct TimeVaryingEdge {
    from: u64,
    to: u64,
    weight: u32,
    time: u64,
}

struct AnnexTimeVaryingGraph {
    max_time: u64,
    nodes: Vec<Node>,
    edges: Vec<TimeVaryingEdge>,
    dst_mat_undel: DistanceMatrix,
    dst_mat_del: Vec<DistanceMatrix>,
}

fn graph_to_temporal(graph: &Graph, max_time: u64, deleted_edges: &Vec<(u64, u64, u64)>) -> AnnexTimeVaryingGraph {
    let mut edges = vec![];
    for edge in &graph.edges {
        for t in 0..max_time {
            let weight = if deleted_edges.contains(&(edge.from, edge.to, t)) {
                u32::MAX
            } else {
                edge.weight
            };
            edges.push(TimeVaryingEdge {
                from: edge.from,
                to: edge.to,
                weight,
                time: t,
            });
        }
    }
    let mut nodes = vec![];
    for edge in &edges {
        if !nodes.contains(&edge.from) {
            nodes.push(edge.from);
        }
        if !nodes.contains(&edge.to) {
            nodes.push(edge.to);
        }
    }
    let dst_mat_undel = from_shortest_paths(&johnson(graph, max_time), &nodes);
    let mut dst_mat_del = vec![];
    for t in 0..max_time {
        let deleted_edges_t: Vec<(u64, u64, u64)> = deleted_edges.iter().filter(|(_, _, time)| *time == t).cloned().collect();
        let graph_t = Graph {
            nodes: nodes.clone(),
            edges: edges.iter().map(|edge| {
                let weight = if deleted_edges_t.contains(&(edge.from, edge.to, t)) {
                    u32::MAX
                } else {
                    edge.weight
                };
                Edge {
                    from: edge.from,
                    to: edge.to,
                    weight,
                }
            }).collect(),
        };
        dst_mat_del.push(from_shortest_paths(&johnson(&graph_t, max_time), &nodes));
    }
    return AnnexTimeVaryingGraph {
        max_time,
        nodes,
        edges,
        dst_mat_undel,
        dst_mat_del,
    };
}

fn print_timegraph(graph: &TimeVaryingGraph) {
    println!("Time graph with max time : {}", graph.max_time);
    // print time by time
    for t in 0..graph.max_time {
        println!("Time {}", t);
        for edge in &graph.edges {
            // if the edge is not deleted at time t, print it
            if !graph.deleted_edges.contains(&(edge.from, edge.to, t)) {
                println!("{} --{}--> {}", edge.from, edge.weight, edge.to);
            }
        }
    }
}

fn print_annex_graph(graph: &AnnexTimeVaryingGraph) {
    println!("Annex graph with max time : {}", graph.max_time);
    // print time by time
    for t in 0..graph.max_time {
        println!("Time {}", t);
        print_matrix(&graph.dst_mat_del[t as usize]);
    }
}

fn recompute_all_distance_matrix(graph: &mut AnnexTimeVaryingGraph) {
    for t in 0..graph.max_time {
        for i in 0..graph.dst_mat_undel.len() {
            for j in 0..graph.dst_mat_undel[i].len() {
                let deleted_edges_t: Vec<(u64, u64, u64)> = graph.edges.iter().filter(|edge| edge.time == t).filter(|edge| edge.weight == u32::MAX).map(|edge| (edge.from, edge.to, edge.time)).collect();
                let graph_t = Graph {
                    nodes: graph.nodes.clone(),
                    edges: graph.edges.iter().map(|edge| {
                        let weight = if deleted_edges_t.contains(&(edge.from, edge.to, t)) {
                            u32::MAX
                        } else {
                            edge.weight
                        };
                        Edge {
                            from: edge.from,
                            to: edge.to,
                            weight,
                        }
                    }).collect(),
                };
                let paths = johnson(&graph_t, graph.max_time);
                graph.dst_mat_del[t as usize] = from_shortest_paths(&paths, &graph.nodes);
            }
        }
    }
}


fn main() {
    let graph = Graph {
        nodes: vec![0, 1, 2],
        edges: vec![
            Edge {
                from: 0,
                to: 1,
                weight: 3,
            },
            Edge {
                from: 2,
                to: 1,
                weight: 1,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 1,
            },
        ],
    };
    println!("Graph:");
    print_graph(&graph);
    println!("Paths:");
    let paths = johnson(&graph);
    for path in &paths {
        print_path(path);
    }
    let matrix = from_shortest_paths(&paths, &graph.nodes);
    println!("Distance matrix:");
    print_matrix(&matrix);

    let time_graph = TimeVaryingGraph {
        max_time: 3,
        edges: vec![
            Edge {
                from: 0,
                to: 1,
                weight: 3,
            },
            Edge {
                from: 2,
                to: 1,
                weight: 1,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 1,
            },
        ],
        deleted_edges: vec![(0, 1, 0), (2, 1, 1), (2, 1, 2)],
    };
    println!("Time graph:");
    print_timegraph(&time_graph);

    let mut annex_graph = graph_to_temporal(&graph, 3, &time_graph.deleted_edges);
    println!("Annex graph:");
    print_annex_graph(&annex_graph);

    recompute_all_distance_matrix(&mut annex_graph);
    println!("Annex graph after recomputation:");
    print_annex_graph(&annex_graph);
}
