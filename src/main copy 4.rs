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
fn dijkstra(graph: &Graph, start: Node, end: Node) -> Path {
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
        if min_node == u64::MAX {
            break;
        }
        visited[min_node as usize] = true;
        for edge in &graph.edges {
            if edge.from == min_node {
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

fn johnson(graph: &Graph) -> Vec<Path> {
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
                let path = dijkstra(graph, *start, *end);
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
    nodes
    edges: Vec<Edge>,
    dst_mat_undel: DistanceMatrix,
    dst_mat_del: Vec<DistanceMatrix>,
}

fn print_timegraph(graph: &TimeVaryingGraph) {
    println!("TVGraph max {}", graph.max_time);
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

fn graph_to_temporal(graph: &Graph, max_time: u64, deleted_edges: &Vec<(u64, u64, u64)>) -> AnnexTimeVaryingGraph {
    // Compute the shortest paths from every node to every other node
    let mut dst_mat_undel = vec![vec![u32::MAX; graph.edges.len()]; graph.edges.len()];
    let paths = johnson(graph);
    for path in &paths {
        if path.steps.is_empty() {
            continue;
        }
        let from = graph.nodes.iter().position(|&n| n == path.from).unwrap();
        let to = graph.nodes.iter().position(|&n| n == path.to).unwrap();
        let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
        dst_mat_undel[from][to] = sum;
    }
    // fill the diagonal with 0
    for i in 0..graph.nodes.len() {
        dst_mat_undel[i][i] = 1;
    }
    let mut dst_mat_del = vec![dst_mat_undel.clone(); max_time as usize];
    // for each deleted edge, check if there was a path that used it
    for (from, to, t) in deleted_edges {
        for path in &paths {
            if path.steps.is_empty() {
                continue;
            }
            let from_idx = graph.nodes.iter().position(|&n| n == path.from).unwrap();
            let to_idx = graph.nodes.iter().position(|&n| n == path.to).unwrap();
            let mut found = false;
            for (i, step) in path.steps.iter().enumerate() {
                if step.0 == *from && step.2 == *to {
                    found = true;
                    println!("Found path using deleted edge {} --{}--> {} at time {}", from, step.1, to, t);
                    break;
                }
            }
            if found {
                dst_mat_del[*t as usize][from_idx][to_idx] = 0;
            }
        }
    }
    return AnnexTimeVaryingGraph {
        max_time,
        dst_mat_undel,
        dst_mat_del,
    };
}

fn temporal_dijkstra(graph: &mut AnnexTimeVaryingGraph, start: Node, dest: Node, time: u64) {
    // If the distance is already computed (=/= 0), return it
    if graph.dst_mat_del[time as usize][start as usize][dest as usize] != 0 {
        println!("Distance from {} to {} at time {} is already computed", start, dest, time);
        return;
    }

    // If the distance is not computed, compute it
    let mut dist = vec![u32::MAX; graph.dst_mat_undel.len()];
    let mut prev = vec![u64::MAX; graph.dst_mat_undel.len()];
    let mut visited = vec![false; graph.dst_mat_undel.len()];
    dist[start as usize] = 0;
    for _ in 0..graph.dst_mat_undel.len() {
        let mut min_dist = u32::MAX;
        let mut min_node = u64::MAX;
        for (i, d) in dist.iter().enumerate() {
            if !visited[i] && *d < min_dist {
                min_dist = *d;
                min_node = i as u64;
            }
        }
        if min_node == u64::MAX {
            break;
        }
        visited[min_node as usize] = true;
        for edge in &graph.dst_mat_undel[min_node as usize] {
            if edge.from == min_node {
                let alt = dist[min_node as usize] + edge.weight;
                if alt < dist[edge.to as usize] {
                    dist[edge.to as usize] = alt;
                    prev[edge.to as usize] = min_node;
                }
            }
        }
    }
    let mut steps = vec![];
    let mut node = dest;
    while node != start {
        let prev_node = prev[node as usize];
        if prev_node == u64::MAX {
            break;
        }
        let weight = dist[node as usize] - dist[prev_node as usize];
        steps.push((prev_node, weight, node));
        node = prev_node;
    }
    // Write the distance in the distance matrix
    let sum = steps.iter().map(|(_, w, _)| w).sum::<u32>();
    if sum == 0 {
        graph.dst_mat_del[time as usize][start as usize][dest as usize] = u32::MAX;
    } else {
        graph.dst_mat_del[time as usize][start as usize][dest as usize] = sum;
    }

}

fn recompute_all_distance_matrix(graph: &mut AnnexTimeVaryingGraph) {
    for t in 0..graph.max_time {
        for i in 0..graph.dst_mat_undel.len() {
            for j in 0..graph.dst_mat_undel[i].len() {
                let path = temporal_dijkstra(graph, i as u64, j as u64, t);
                if path.steps.is_empty() {
                    graph.dst_mat_del[t as usize][i][j] = u32::MAX;
                } else {
                    let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
                    graph.dst_mat_del[t as usize][i][j] = sum;
                }
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
        deleted_edges: vec![(0, 1, 0), (0, 1, 1)],
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
