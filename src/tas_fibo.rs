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

fn dijkstra(graph: &Graph, start: Node, end: Node, max_time: u64, distance_matrix: &mut DistanceMatrix) -> Path {
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
            distance_matrix[start as usize][node as usize] = cost;
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


fn johnson(graph: &Graph, max_time: u64) -> Vec<Path> {
    let mut paths = vec![];
    let mut distance_matrix = vec![vec![u32::MAX; graph.max_node_index as usize]; graph.max_node_index as usize];
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                distance_matrix[i as usize][j as usize] = 0;
            }
        }
    }
    for edge in &graph.edges {
        distance_matrix[edge.from as usize][edge.to as usize] = edge.weight;
    }
    for k in 0..graph.max_node_index {
        for i in 0..graph.max_node_index {
            for j in 0..graph.max_node_index {
                if distance_matrix[i as usize][k as usize] + distance_matrix[k as usize][j as usize] < distance_matrix[i as usize][j as usize] {
                    distance_matrix[i as usize][j as usize] = distance_matrix[i as usize][k as usize] + distance_matrix[k as usize][j as usize];
                }
            }
        }
    }
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                continue;
            }
            if distance_matrix[i as usize][j as usize] != u32::MAX {
                let path = dijkstra(graph, i, j, max_time, &mut distance_matrix);
                paths.push(path);
            }
        }
    }
    return paths;
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

fn graph_to_temporal(graph: &Graph, max_time: u64, deleted_edges: &Vec<(u64, u64, u64)>) -> (AnnexTimeVaryingGraph, Vec<(u64, u64, u64)>) {
    let nodes = graph.nodes.clone();
    // compute the distance matrix for the original graph
    let mut paths = johnson(graph, max_time);
    let dst_mat_undel = from_shortest_paths(&paths, &nodes, graph.max_node_index, max_time);
    let mut dst_mat_del = vec![dst_mat_undel.clone(); max_time as usize];

    // TODO : fix dependencies not calculating correctly i think
    // Cause some paths may be affected but not detected.
    let mut annex_edges = vec![];
    for edge in &graph.edges {
        let mut weights = vec![];
        for t in 0..max_time {
            if deleted_edges.contains(&(edge.from, edge.to, t)) {
                // get wait time since
                let mut wait_time = 1;
                while deleted_edges.contains(&(edge.from, edge.to, t + wait_time)) {
                    wait_time += 1;
                }
                //let weight_at_t = get_weight2(edge, t);
                weights.push(get_weight2(edge, t + wait_time) + wait_time as u32);
                for i in t..=t + wait_time {
                    dst_mat_del[i as usize][edge.from as usize][edge.to as usize] = 0;
                }
                //nodes_to_do_dijkstra.push((edge.from, edge.to, t));
            } else {
                weights.push(get_weight2(edge, t));
            }
        }
        annex_edges.push(TimeVaryingEdge {
            from: edge.from,
            to: edge.to,
            weight: weights,
        });
    }
    for path in &paths {
        if path.steps.is_empty() {
            continue;
        }
        // calculate if the path was perturbed
        let mut perturbed_at = vec![];
        let mut perturbed = false;
        let mut sum = 0;
        let mut current_node = path.from;
        let mut next_node;
        for (from, weight, to) in &path.steps {
            next_node = *to;
            sum += weight;
            if deleted_edges.contains(&(current_node, next_node, sum as u64)) {
                perturbed = true;
                perturbed_at.push(sum);
            }
            current_node = *to;
        }
        // if perturbed, reset all the distances written by the path where the perturbation would make them obsolete
        if perturbed {
            for t in 0..max_time {
                for i in 0..dst_mat_del[t as usize].len() {
                    for j in 0..dst_mat_del[t as usize][i].len() {
                        if i == j {
                            continue;
                        }
                        let distance = dst_mat_del[t as usize][i][j];
                        if distance != u32::MAX && distance != 0 {
                            for perturbation in &perturbed_at {
                                if distance + *perturbation as u32 > max_time as u32 {
                                    dst_mat_del[t as usize][i][j] = u32::MAX;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let annex_graph = AnnexTimeVaryingGraph {
        max_time,
        nodes,
        edges: annex_edges,
        dst_mat_undel,
        dst_mat_del,
    };
    (annex_graph, vec![])

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

fn temporal_dijkstra(graph: &AnnexTimeVaryingGraph, start: Node, end: Node, max_time: u64, time: u64, max_node_index: u64, distance_matrix: &mut DistanceMatrix) -> Path {
    let mut dist = vec![u32::MAX; max_node_index as usize];
    let mut prev = vec![u64::MAX; max_node_index as usize];
    let mut visited = vec![false; max_node_index as usize];
    dist[start as usize] = 0;
    let mut heap = Heap::new();
    heap.push(DijkstraNode {
        node: start,
        cost: 0,
    });
    while let Some(DijkstraNode { node, cost }) = heap.pop() {
        if visited[node as usize] {
            distance_matrix[start as usize][node as usize] = cost;
            continue;
        }
        visited[node as usize] = true;
        for edge in &graph.edges {
            if edge.from == node {
                if dist[node as usize] == u32::MAX {
                    continue;
                }
                let alt = dist[node as usize] + edge.weight[time as usize];
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
    // write the distance from node start to every node at time t
    let mut sum = 0;
    for (from, weight, to) in &steps {
        sum += *weight;
        if sum + time as u32 > max_time as u32 {
            return Path {
                from: start,
                to: end,
                steps: vec![],
            };
        }
        if distance_matrix[start as usize][*to as usize] != 0 {
            if distance_matrix[start as usize][*to as usize] > sum {
                distance_matrix[start as usize][*to as usize] = sum;
            }
        } else {
            distance_matrix[start as usize][*to as usize] = sum;
        }
    }
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
                    println!("Recomputing distance from {} to {} at time {}", i, j, t);
                    // TODO : take adventage of the fact that we already have the shortest path from a bunch of other nodes
                    let dst_matrix = unsafe { std::mem::transmute::<&mut DistanceMatrix, &mut DistanceMatrix>(&mut graph.dst_mat_del[t as usize]) };
                    let path = temporal_dijkstra(graph, i as u64, j as u64, graph.max_time, t, max_node_index, dst_matrix);
                    print_path(&path);
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
                        println!("{} --{}-> {}, sum : {}", from, weight, to, sum);
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
    let edges : Vec<Edge> = (0..200).map(|i| {
        Edge {
            from: i,
            to: i + 1,
            weight: 1,
        }
    }).collect();
    let deleted_edges = vec![(0, 1, 0), (0, 1, 1)];
    let nodes = (0..201).map(|i| (i, vec![(i + 1, 1)])).collect();
    //let max_node_index = 201;
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
    let max_time = 3;
    */

    let max_node_index = edges.iter().map(|e| e.from.max(e.to)).max().unwrap() + 1;
    let graph = Graph {
        max_node_index,
        nodes,
        edges,
    };
    let time_graph = TimeVaryingGraph {
        max_time,
        edges : graph.edges.clone(),
        deleted_edges : deleted_edges.clone(),
    };

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
