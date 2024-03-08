type Node = u64;
type Weight = u32;

use std::time;

use fibonacii_heap::Heap;


macro_rules! benchmark {
    ($name:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        println!("Time elapsed in {}() is: {:?}", $name, duration);
        result
    }};
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeletedLink {
    from: Node,
    to: Node,
    times: Vec<u64>,
}

struct DeletedLinksMatrix {
    links: Vec<Vec<Vec<u64>>>
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct PathSlice {
    index_start: usize,
    index_end: usize,
    time_start: u64,
    total_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

// TODO : fix that some subpaths are from the same node to the same node
/*fn get_correlated_paths(path: &Path, max_time: u64, dst_mat: &DistanceMatrix) -> CorrelatedPaths {
    
    // Path = ((0--2-->1), (1--1-->2), (2--1-->3))
    // Create {(0--4-->3), (0--3-->2), (0--2-->1), (1--2-->3), (1--1-->2), (2--1-->3)}

    let mut pairs = vec![];
    let mut sums = vec![];
    let mut total_length = 0;
    for i in 0..path.steps.len() {
        let (from, weight, to) = &path.steps[i];
        total_length += weight;
        let mut sum = 0;
        for j in i..path.steps.len() {
            let (from2, weight2, to2) = &path.steps[j];
            sum += weight2;
            if sum <= max_time as u32 {
                pairs.push((*from, *to2));
                sums.push(sum);
            }
        }
    }
    println!("Pairs : {:?}", pairs);
    println!("Sums : {:?}", sums);
    let mut subpaths = vec![];
    for i in 0..pairs.len() {
        let (from, to) = pairs[i];
        let sum = sums[i];
        let mut index_start = 0;
        let mut index_end = 0;
        for j in 0..path.steps.len() {
            let (from2, weight, to2) = &path.steps[j];
            if *from2 == from {
                index_start = j;
            }
            if *to2 == to {
                index_end = j;
            }
        }
        subpaths.push(PathSlice {
            index_start,
            index_end,
            time_start: sum as u64,
            total_length,
        });
    }
    return CorrelatedPaths {
        main_path: path.clone(),
        subpaths,
    };

}*/

struct SubPath {
    from: Node,
    to: Node,
    time_start: u64,
    total_length: u32,

}

// Takes 160us for a path of 200 steps, so very very good, considering that its not optimized at all
// And all it gives us (n² - n) / 2 paths, so it's worth it i think
/*fn get_correlated_paths(path: &Path, max_time: u64, dst_mat: &DistanceMatrix) -> Vec<SubPath> {
    let mut pairs = vec![];
    let mut sums = vec![];
    let mut lengths = vec![];
    let mut total_lengths = vec![];
    let mut times = vec![];
    let mut total_length = 0;
    benchmark!("making pairs",
    for i in 0..path.steps.len() {
        let (from, weight, to) = &path.steps[i];
        total_length += weight;
        let mut sum = 0;
        for j in i..path.steps.len() {
            let (from2, weight2, to2) = &path.steps[j];
            sum += weight2;
            if sum <= max_time as u32 {
                pairs.push((*from, *to2));
                sums.push(sum);
                lengths.push(j - i + 1);
                total_lengths.push(total_length);
                times.push(i as u64);
            }
        }
    });
    
    let mut paths = pairs.iter().zip(sums.iter()).zip(lengths.iter()).zip(total_lengths.iter()).zip(times.iter()).collect::<Vec<_>>();
    for i in 0..paths.len() {
        let (((((from, to), sum), length), total_length), time) = paths[i];
        println!("{} --{}-> {} at start time {}", from, sum, to, time);
    }

    let mut subpaths = vec![];
    for i in 0..pairs.len() {
        let (from, to) = pairs[i];
        let sum = sums[i];
        let length = lengths[i];
        let total_length = total_lengths[i];
        let time = times[i];
        let mut index_start = 0;
        let mut index_end = 0;
        for j in 0..path.steps.len() {
            let (from2, weight, to2) = &path.steps[j];
            if *from2 == from {
                index_start = j;
            }
            if *to2 == to {
                index_end = j;
            }
        }
        subpaths.push(PathSlice {
            index_start,
            index_end,
            time_start: sum as u64,
            total_length,
        });
    }
    return CorrelatedPaths {
        main_path: path.clone(),
        subpaths,
    };
}*/

fn get_correlated_paths(path: &Path, max_time: u64, dst_mat: &DistanceMatrix) -> Vec<SubPath> {
    let mut pairs = vec![];
    let mut sums = vec![];
    let mut lengths = vec![];
    let mut total_lengths = vec![];
    let mut times = vec![];
    let mut total_length = 0;
    benchmark!("making pairs",
    for i in 0..path.steps.len() {
        let (from, weight, to) = &path.steps[i];
        total_length += weight;
        let mut sum = 0;
        for j in i..path.steps.len() {
            let (from2, weight2, to2) = &path.steps[j];
            sum += weight2;
            if sum <= max_time as u32 {
                pairs.push((*from, *to2));
                sums.push(sum);
                lengths.push(j - i + 1);
                total_lengths.push(total_length);
                times.push(i as u64);
            }
        }
    });
    
    let mut paths = pairs.iter().zip(sums.iter()).zip(lengths.iter()).zip(total_lengths.iter()).zip(times.iter()).collect::<Vec<_>>();
    /*for i in 0..paths.len() {
        let (((((from, to), sum), length), total_length), time) = paths[i];
        println!("{} --{}-> {} at start time {}", from, sum, to, time);
    }*/

    let mut subpaths = vec![];
    for i in 0..pairs.len() {
        let (from, to) = pairs[i];
        let sum = sums[i];
        let length = lengths[i];
        let total_length = total_lengths[i];
        let time = times[i];
        println!("{} --{}-> {} at start time {}", from, sum, to, time);
        subpaths.push(SubPath {
            from: from,
            to: to,
            time_start: sum as u64,
            total_length: total_length,
        });
    }
    
    return subpaths;
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
fn from_shortest_corr_paths(paths: &Vec<CorrelatedPaths>, nodes: &Vec<(Node, Neighbours)>, max_node_index: u64, max_time: u64, deleted_edges: &Vec<DeletedLink>) -> DistanceMatrix {
    let mut original_matrix = vec![vec![u32::MAX; max_node_index as usize]; max_node_index as usize];
    // Fill the diagonal with 0
    for i in 0..max_node_index {
        original_matrix[i as usize][i as usize] = 1;
    }
    for path in paths {
        if path.main_path.steps.is_empty() {
            continue;
        }
        let from = nodes.iter().position(|&(n, _)| n == path.main_path.from).unwrap();
        let to = nodes.iter().position(|&(n, _)| n == path.main_path.to).unwrap();
        let sum = path.main_path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
        if sum <= max_time as u32 {
            original_matrix[from][to] = sum;
        }
    }

    return original_matrix;
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


fn create_phantom_edges(graph: &Graph, max_time: u64, deleted_edges: &Vec<DeletedLink>, dst_mat_del: &mut Vec<DistanceMatrix>) -> Vec<TimeVaryingEdge> {
    let mut edges = vec![];
    for deleted_link in deleted_edges {
        for t in 0..max_time {
            if deleted_link.times.contains(&t) {
                continue;
            }
            let mut weight = vec![u32::MAX; max_time as usize];
            for t2 in 0..max_time {
                if t2 == t {
                    weight[t2 as usize] = 0;
                } else {
                    weight[t2 as usize] = u32::MAX;
                }
            }
            edges.push(TimeVaryingEdge {
                from: deleted_link.from,
                to: deleted_link.to,
                weight,
            });
            dst_mat_del[t as usize][deleted_link.from as usize][deleted_link.to as usize] = 0;
        }
    }
    return edges;
}



// Unvalidates all shortest paths that go through a deleted edge
// By writing 0 in the distance matrix (0 means it has to be recomputed)
// Infinity means that there is no path, but not that it has to be recomputed
fn invalidate_deleted_edges(paths: &Vec<CorrelatedPaths>, deleted_edges: &DeletedLinksMatrix, max_time: u64, dst_mat_del: &mut Vec<DistanceMatrix>) {
    for path in paths {
        let backtrace = path.main_path.steps.len() > 15;
        let main_path = &path.main_path;
        let mut sum = 0;
        for i in 0..main_path.steps.len() {
            let (from, weight, to) = &main_path.steps[i];
            // see if the edge was deleted at sum
            if deleted_edges.links[*from as usize][*to as usize].contains(&sum) {
                // invalidate the distance matrix
                dst_mat_del[sum as usize][*from as usize][*to as usize] = 0;
                // for the rest of the path, we have to invalidate the distance matrix
                for j in i+1..main_path.steps.len() {
                    /*if backtrace {
                        println!("Invalidating path from {} to {} at time {} with sum {}", main_path.from, main_path.to, 0, sum);
                    }*/
                    let (node_from, weight, node_to) = &main_path.steps[j];
                    sum += *weight as u64;
                    /*if sum >= max_time {
                        break;
                    }
                    dst_mat_del[sum as usize][*node_from as usize][*node_to as usize] = 0;*/
                    if dst_mat_del[0][*from as usize][*node_to as usize] != u32::MAX {
                        dst_mat_del[0][*from as usize][*node_to as usize] = 0;
                    }
                }
            }
            sum += *weight as u64;
            //println!("{} --{}-> {}", from, weight, to);
        }
    }

}

fn graph_to_temporal(graph: &Graph, max_time: u64, deleted_edges: &Vec<DeletedLink>) -> (AnnexTimeVaryingGraph, Vec<(u64, u64, u64)>) {
    let mut edges : Vec<TimeVaryingEdge> = vec![];
    let (paths,_) = benchmark!("johnson", johnson(graph, max_time));
    let mut dst_mat_undel = benchmark!("from_corr_paths", from_shortest_corr_paths(&paths, &graph.nodes, graph.max_node_index, max_time, deleted_edges));
    print_matrix(&dst_mat_undel);
    // TODO : change that to infinity once debuggin is done
    let mut dst_mat_del = vec![vec![vec![1; graph.max_node_index as usize]; graph.max_node_index as usize]; max_time as usize];

    let annex_edges = benchmark!("create_phantom_edges", create_phantom_edges(&graph, max_time, &deleted_edges, &mut dst_mat_del));

    // 0 if directly deleted
    /*for deleted_link in deleted_edges {
        for t in &deleted_link.times {
            dst_mat_del[*t as usize][deleted_link.from as usize][deleted_link.to as usize] = 0;
        }
    }*/
    
    // TODO : yeah this is very space consuming, probably better to switch to a sparse matrix or hashmap
    let deleted_edges_matrix = benchmark!("deleted_edges_matrix", {
        let max_node_index = graph.nodes.iter().map(|(n, _)| n).max().unwrap() + 1;
        let mut deleted_edges_matrix = vec![vec![vec![]; max_node_index as usize]; max_node_index as usize];
        for deleted_link in deleted_edges {
            for t in &deleted_link.times {
                deleted_edges_matrix[deleted_link.from as usize][deleted_link.to as usize].push(*t);
            }
        }
        let deleted_edges_matrix = DeletedLinksMatrix {
            links: deleted_edges_matrix,
        };
        deleted_edges_matrix
    });
    
    
    benchmark!("invalidate_deleted_edges", invalidate_deleted_edges(&paths, &deleted_edges_matrix, max_time, &mut dst_mat_del));
    println!("Deleted distance matrices 0");
    print_matrix(&dst_mat_del[0]);
    println!("Deleted distance matrices 1");
    print_matrix(&dst_mat_del[1]);
    println!("Deleted distance matrices 2");
    print_matrix(&dst_mat_del[2]);
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
        edges: annex_edges,
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
    /*
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
    let deleted_edges = vec![DeletedLink {
        from: 0,
        to: 1,
        times: vec![0, 1],
    }];
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
    println!("Reachables : {}", reachables);*/

    let from = 0;
    let to = 20;
    let steps = (from..to).map(|i| (i, 1, i + 1)).collect();
    let path = Path {
        from,
        to,
        steps,
    };
    let correlated_paths = get_correlated_paths(&path, 20, &vec![]);
}
