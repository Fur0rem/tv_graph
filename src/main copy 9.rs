type Node = u64;
type Weight = u32;

use fibonacii_heap::Heap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Edge {
    from: Node,
    to: Node,
    weight: Weight,
}
type Neighbours = Vec<(Node, Weight)>;
struct Graph {
    max_node_index: u64,
    nodes: Vec<(Node, Neighbours)>,
    edges: Vec<Edge>,
}

fn print_graph(graph: &Graph) {
    for edge in &graph.edges {
        println!("{} --{}--> {}", edge.from, edge.weight, edge.to);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
struct DijkstraNode {
    node: Node,
    cost: Weight,
}

impl PartialEq for DijkstraNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for DijkstraNode {}

impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.cmp(&other.cost)
    }
}

/*fn dijkstra(graph: &Graph, start: Node, end: Node, max_time: u64) -> Path {
    let mut dist = vec![u32::MAX; graph.max_node_index as usize];
    let mut prev = vec![u64::MAX; graph.max_node_index as usize];
    let mut visited = vec![false; graph.max_node_index as usize];
    dist[start as usize] = 0;
    let mut heap = Heap::new();
    heap.push(DijkstraNode {
        node: start,
        cost: 0,
    });
    while let Some(DijkstraNode { node, cost }) = heap.pop() {
        if visited[node as usize] {
            continue;
        }
        visited[node as usize] = true;
        for edge in &graph.edges {
            if edge.from == node {
                if dist[node as usize] == u32::MAX {
                    continue;
                }
                let alt = dist[node as usize] + edge.weight;
                if alt < dist[edge.to as usize] {
                    dist[edge.to as usize] = alt;
                    prev[edge.to as usize] = node;
                    heap.push(DijkstraNode {
                        node: edge.to,
                        cost: alt,
                    });
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
}*/

struct PathSlice {
    index_start: usize,
    index_end: usize,
    time_start: u64,
    total_length: u32,
}

struct CorrelatedPaths {
    main_path: Path,
    subpaths: Vec<PathSlice>,
}

fn dijkstra(graph: &Graph, start: Node, end: Node, max_time: u64, dist_mat: &mut DistanceMatrix) -> Path {
    let mut dist = vec![u32::MAX; graph.max_node_index as usize];
    let mut prev = vec![u64::MAX; graph.max_node_index as usize];
    let mut visited = vec![false; graph.max_node_index as usize];
    dist[start as usize] = 0;
    let mut heap = Heap::new();
    heap.push(DijkstraNode {
        node: start,
        cost: 0,
    });
    while let Some(DijkstraNode { node, cost }) = heap.pop() {
        if visited[node as usize] {
            continue;
        }
        visited[node as usize] = true;
        for edge in &graph.edges {
            if edge.from == node {
                if dist[node as usize] == u32::MAX {
                    continue;
                }
                let alt = dist[node as usize] + edge.weight;
                if alt < dist[edge.to as usize] {
                    dist[edge.to as usize] = alt;
                    prev[edge.to as usize] = node;
                    heap.push(DijkstraNode {
                        node: edge.to,
                        cost: alt,
                    });
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

fn get_correlated_paths(path: &Path, max_time: u64, dst_mat: &DistanceMatrix) -> CorrelatedPaths {
    let mut subpaths = vec![];
    let mut index_start = 0;
    let mut index_end = 0;
    let mut time_start = 0;
    let mut total_length = 0;
    for (i, (from, weight, to)) in path.steps.iter().enumerate() {
        if total_length + weight > max_time as u32 {
            subpaths.push(PathSlice {
                index_start,
                index_end,
                time_start,
                total_length,
            });
            index_start = i;
            index_end = i;
            time_start = total_length as u64;
            total_length = *weight as u32;
        } else {
            index_end = i;
            total_length += weight;
        }
    }
    subpaths.push(PathSlice {
        index_start,
        index_end,
        time_start,
        total_length,
    });
    return CorrelatedPaths {
        main_path: path.clone(),
        subpaths,
    };
}

fn johnson(graph: &Graph, max_time: u64) -> (Vec<CorrelatedPaths>, DistanceMatrix) {
    let mut paths = vec![];
    let mut dst_mat = vec![vec![u32::MAX; graph.max_node_index as usize]; graph.max_node_index as usize];
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                dst_mat[i as usize][j as usize] = 1;
            }
        }
    }
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                continue;
            }
            let path = dijkstra(graph, i, j, max_time, &mut dst_mat);
            let correlated_paths = get_correlated_paths(&path, max_time, &dst_mat);
            paths.push(correlated_paths);
        }
    }
    return (paths, dst_mat);
}

type DistanceMatrix = Vec<Vec<u32>>;

fn from_shortest_paths(paths: &Vec<Path>, nodes: &Vec<(Node, Neighbours)>, max_node_index: u64, max_time: u64) -> DistanceMatrix {
    let mut matrix = vec![vec![u32::MAX; max_node_index as usize]; max_node_index as usize];
    // Fill the diagonal with 0
    for i in 0..max_node_index {
        matrix[i as usize][i as usize] = 1;
    }
    for path in paths {
        if path.steps.is_empty() {
            continue;
        }
        let from = nodes.iter().position(|&(n, _)| n == path.from).unwrap();
        let to = nodes.iter().position(|&(n, _)| n == path.to).unwrap();
        let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
        if sum <= max_time as u32 {
            matrix[from][to] = sum;
        }
    }
    return matrix;
}

/*fn from_shortest_corr_paths(paths: &Vec<CorrelatedPaths>, nodes: &Vec<(Node, Neighbours)>, max_node_index: u64, max_time: u64) -> DistanceMatrix {
    let mut matrix = vec![vec![u32::MAX; max_node_index as usize]; max_node_index as usize];
    // Fill the diagonal with 0
    for i in 0..max_node_index {
        matrix[i as usize][i as usize] = 1;
    }
    for path in paths {
        if path.main_path.steps.is_empty() {
            continue;
        }
        let from = nodes.iter().position(|&(n, _)| n == path.main_path.from).unwrap();
        let to = nodes.iter().position(|&(n, _)| n == path.main_path.to).unwrap();
        let sum = path.main_path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
        if sum <= max_time as u32 {
            matrix[from][to] = sum;
        }
    }
    return matrix;
}*/

// Returns the distance matrix of the unattacked graph and the other distance matrices
fn from_shortest_corr_paths(paths: &Vec<CorrelatedPaths>, nodes: &Vec<(Node, Neighbours)>, max_node_index: u64, max_time: u64, deleted_edges: &Vec<(u64, u64, u64)>) -> (DistanceMatrix, Vec<DistanceMatrix>) {
    let mut matrix = vec![vec![u32::MAX; max_node_index as usize]; max_node_index as usize];
    let mut matrices = vec![];
    // Fill the diagonal with 0
    for i in 0..max_node_index {
        matrix[i as usize][i as usize] = 1;
    }
    for path in paths {
        if path.main_path.steps.is_empty() {
            continue;
        }
        let from = nodes.iter().position(|&(n, _)| n == path.main_path.from).unwrap();
        let to = nodes.iter().position(|&(n, _)| n == path.main_path.to).unwrap();
        let sum = path.main_path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
        if sum <= max_time as u32 {
            matrix[from][to] = sum;
        }
    }
    matrices.push(matrix.clone());
    for (i, edge) in deleted_edges.iter().enumerate() {
        let mut matrix = matrix.clone();
        for (i, row) in matrix.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                if *cell == u32::MAX {
                    continue;
                }
                if *cell + edge.2 as u32 > max_time as u32 {
                    matrix[i][j] = u32::MAX;
                }
            }
        }
        matrices.push(matrix);
    }
    return (matrix, matrices);
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
    weight: Vec<u32>,
}

struct AnnexTimeVaryingGraph {
    max_time: u64,
    nodes: Vec<(Node, Neighbours)>,
    edges: Vec<TimeVaryingEdge>,
    dst_mat_undel: DistanceMatrix,
    dst_mat_del: Vec<DistanceMatrix>,
}

fn get_weight2(edge: &Edge, t: u64) -> u32 {
    return edge.weight;
}

macro_rules! benchmark {
    ($name:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        println!("Time elapsed in {}() is: {:?}", $name, duration);
        result
    }};
}


fn graph_to_temporal(graph: &Graph, max_time: u64, deleted_edges: &Vec<(u64, u64, u64)>) -> (AnnexTimeVaryingGraph, Vec<(u64, u64, u64)>) {
    let mut edges = vec![];
    let mut paths = benchmark!("johnson", johnson(graph, max_time));
    let (mut dst_mat_undel, dst_mat_del) = benchmark!("from_corr_paths", from_shortest_corr_paths(&paths.0, &graph.nodes, graph.max_node_index, max_time));
    print_matrix(&dst_mat_undel);
    let mut dst_mat_del = vec![vec![vec![u32::MAX; graph.max_node_index as usize]; graph.max_node_index as usize]; max_time as usize];
    /*for (i, edge) in graph.edges.iter().enumerate() {
        let mut weight = vec![u32::MAX; max_time as usize];
        for t in 0..max_time {
            if deleted_edges.contains(&(edge.from, edge.to, t)) {
                weight[t as usize] = 0;<
                dst_mat_del[t as usize][edge.from as usize][edge.to as usize] = 0;
            } else {
                weight[t as usize] = get_weight2(edge, t);
            }
        }
        edges.push(TimeVaryingEdge {
            from: edge.from,
            to: edge.to,
            weight,
        });
    }
    let mut todo = vec![];
    for t in 0..max_time {
        for i in 0..dst_mat_undel.len() {
            for j in 0..dst_mat_undel[i].len() {
                if dst_mat_undel[i][j] == u32::MAX {
                    todo.push((t, i as u64, j as u64));
                }
            }
        }
    }*/
    return (AnnexTimeVaryingGraph {
        max_time,
        nodes: graph.nodes.clone(),
        edges,
        dst_mat_undel,
        dst_mat_del,
    }, vec![]);

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

// TODO : optimize to write every min distance computed
fn temporal_dijkstra(graph: &AnnexTimeVaryingGraph, start: Node, end: Node, max_time: u64, time: u64, max_node_index: u64) -> Path {
    let mut dist = vec![u32::MAX; max_node_index as usize];
    let mut prev = vec![u64::MAX; max_node_index as usize];
    let mut visited = vec![false; max_node_index as usize];
    dist[start as usize] = 0;
    for _ in 0..max_node_index {
        let mut min_dist = u32::MAX;
        let mut min_node = u64::MAX;
        for (i, d) in dist.iter().enumerate() {
            if !visited[i] && *d < min_dist {
                min_dist = *d;
                min_node = i as u64;
            }
        }
        /*if min_node > max_time {
            break;
        }*/
        if min_node == u64::MAX {
            break;
        }
        visited[min_node as usize] = true;
        for edge in &graph.edges {
            if edge.from == min_node {
                if dist[min_node as usize] == u32::MAX || edge.weight[time as usize] == u32::MAX {
                    continue;
                }
                let alt = dist[min_node as usize] + edge.weight[time as usize];
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

fn recompute_all_distance_matrix(graph: &mut AnnexTimeVaryingGraph) {
    println!("Recomputing all distance matrix, max time : {}", graph.max_time);
    let max_node_index = graph.nodes.iter().map(|(n, _)| n).max().unwrap() + 1;
    for t in 0..graph.max_time {
        for i in 0..graph.dst_mat_undel.len() {
            for j in 0..graph.dst_mat_undel[i].len() {
                if graph.dst_mat_del[t as usize][i][j] == 0 {
                    //println!("Recomputing distance from {} to {} at time {}", i, j, t);
                    // TODO : take adventage of the fact that we already have the shortest path from a bunch of other nodes
                    let path = temporal_dijkstra(graph, i as u64, j as u64, graph.max_time, t, max_node_index);
                    //
                    
                    if path.from == 0 && t == 0 {
                        print_path(&path);
                    }
                    /*let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
                    // TODO TAG PROUT
                    if sum + t as u32 > graph.max_time as u32 {
                        graph.dst_mat_del[t as usize][i][j] = u32::MAX;
                        continue;
                    }
                    graph.dst_mat_del[t as usize][i][j] = sum;*/
                    if path.steps.is_empty() {
                        graph.dst_mat_del[t as usize][i][j] = u32::MAX;
                        continue;
                    }
                    let mut sum = 0;
                    for (from, weight, to) in &path.steps {
                        sum += *weight;
                        //println!("{} --{}-> {}, sum : {}", from, weight, to, sum);
                        if sum + t as u32 >= graph.max_time as u32 {
                            graph.dst_mat_del[t as usize][i][j] = u32::MAX;
                            continue;
                        }
                        graph.dst_mat_del[sum as usize + t as usize][path.from as usize][*to as usize] = sum as u32;
                    }
                    graph.dst_mat_del[t as usize][path.from as usize][path.to as usize] = sum as u32;
                    
                }
                // Remove the distance if it is too high
                // TODO TAG PROUT
                if graph.dst_mat_del[t as usize][i][j] == u32::MAX {
                    continue;
                }
                if graph.dst_mat_del[t as usize][i][j] + t as u32 > graph.max_time as u32 {
                    graph.dst_mat_del[t as usize][i][j] = u32::MAX;
                }
            }
        }
        println!("Time {}", t);
    }
}

fn sum_dma(dma : &Vec<DistanceMatrix>, max_time: u64) -> (f64, u64) {
    let mut sum = 0.0;
    let mut reachables = 0;
    let distances_len = dma[0 as usize].len();
    for t in 0..dma.len() {
        for i in 0..distances_len {
            for j in 0..distances_len {
                if i == j {
                    continue;
                }
                let distance = dma[t as usize][i as usize][j as usize];
                if distance != u32::MAX && distance != 0 {
                    sum += 1.0 / (distance as f64);
                    reachables += 1;
                }
            }
        }
    }
    sum -= distances_len as f64;
    sum /= dma.len() as f64 * (distances_len * (distances_len - 1)) as f64;
    (sum, reachables)
}
/*fn main() {
    let edges = vec![
            Edge {
                from: 0,
                to: 1,
                weight: 1,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 1,
            },
            Edge {
                from: 2,
                to: 1,
                weight: 1,
            },
        ];    
    let deleted_edges = vec![(0, 1, 0), (0, 1, 1)];
    let nodes = vec![0, 1, 2];
    let graph = Graph {
        max_node_index: 3,
        nodes,
        edges,
    };
    let time_graph = TimeVaryingGraph {
        max_time: 3,
        edges : graph.edges.clone(),
        deleted_edges : deleted_edges.clone(),
    };

    let (mut annex_graph,_) = graph_to_temporal(&graph, 3, &deleted_edges);
    print_annex_graph(&annex_graph);
    // print_graph(&graph);
    // print_timegraph(&time_graph);
    // print_annex_graph(&annex_graph);

    let start = std::time::Instant::now();
    recompute_all_distance_matrix(&mut annex_graph);
    let duration = start.elapsed();
    println!("Time elapsed in recompute_all_distance_matrix() is: {:?}", duration);
    let (sum, reachables) = sum_dma(&annex_graph.dst_mat_del, 3);
    println!("Sum of 1/distance is : {}", sum);
    println!("Reachables : {}", reachables);
    print_annex_graph(&annex_graph);
}*/


fn main() {
    /*let nb_nodes = 500;
    let edges : Vec<Edge> = (0..nb_nodes).map(|i| {
        Edge {
            from: i,
            to: i + 1,
            weight: 1,
        }
    }).collect();
    let deleted_edges = vec![(0, 1, 0), (0, 1, 1), (5, 6, 0), (7, 8, 14), (8, 9, 14), (9, 10, 8), (10, 11, 2), (11, 12, 0), (22, 23, 0), (55, 56, 12), (56, 57, 12), (57, 58, 12), (58, 59, 12), (59, 60, 12), (60, 61, 12), (61, 62, 12), (62, 63, 12), (63, 64, 12), (64, 65, 12), (65, 66, 12), (66, 67, 12), (67, 68, 12), (68, 69, 45), (69, 70, 12), (70, 71, 73), (71, 72, 112), (72, 73, 12), (73, 74, 12), (74, 75, 12)];
    let nodes = (0..nb_nodes+1).map(|i| (i, vec![(i + 1, 1)])).collect();
    let max_time = 200;*/

    let nb_nodes = 200;
    let edges : Vec<Edge> = (0..nb_nodes).map(|i| {
        Edge {
            from: i,
            to: i + 1,
            weight: 1,
        }
    }).collect();
    let deleted_edges = vec![(0, 1, 0), (0, 1, 1)];
    let nodes = (0..nb_nodes+1).map(|i| (i, vec![(i + 1, 1)])).collect();
    let max_time = 20;

    /*let edges = vec![
            Edge {
                from: 0,
                to: 1,
                weight: 3,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 1,
            },
            Edge {
                from: 2,
                to: 1,
                weight: 1,
            },
        ];
    let deleted_edges = vec![(0, 1, 0), (0, 1, 1)];
    let nodes = vec![(0, vec![(1, 3), (2, 1)]), (1, vec![]), (2, vec![(1, 1)])];
    let max_time = 3;*/
    let max_node_index = edges.iter().map(|e| e.from.max(e.to)).max().unwrap() + 1;
    let graph = Graph {
        max_node_index,
        nodes,
        edges,
    };
    /*let time_graph = TimeVaryingGraph {
        max_time,
        edges : graph.edges.clone(),
        deleted_edges : deleted_edges.clone(),
    };*/

    let (mut annex_graph, todo) = graph_to_temporal(&graph, max_time, &deleted_edges);
    //print_graph(&graph);
    // print_timegraph(&time_graph);
    //print_annex_graph(&annex_graph);

    let start = std::time::Instant::now();
    recompute_all_distance_matrix(&mut annex_graph);
    let duration = start.elapsed();
    println!("Time elapsed in recompute_all_distance_matrix() is: {:?}", duration);
    let (sum, reachables) = sum_dma(&annex_graph.dst_mat_del, 20);
    println!("Sum of 1/distance is : {}", sum);
    println!("Reachables : {}", reachables);
    print_annex_graph(&annex_graph);
}
