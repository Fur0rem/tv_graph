type Node = u64;
type Weight = u32;

use std::{ops::Sub, time};

use fibonacii_heap::Heap;
use rayon::vec;

use primitive_types::U256;

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

struct DeletedLinkBitMask {
    from: Node,
    to: Node,
    times: U256,
}

struct DeletedLinksMatrix {
    links: Vec<Vec<U256>>
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
fn old_get_correlated_paths(path: &Path, max_time: u64, dst_mat: &DistanceMatrix) -> CorrelatedPaths {
    
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
    //println!("Pairs : {:?}", pairs);
    //println!("Sums : {:?}", sums);
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

}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubPath {
    from: Node,
    to: Node,
    time_start: u64,
    total_length: u32,
    invalidated_at: U256,
}

struct CorrelatedPathsTwo {
    main_path: Path,
    subpaths: Vec<SubPath>,
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

fn get_correlated_paths(path: &Path, max_time: u64) -> Vec<SubPath> {
    let mut pairs = vec![];
    let mut sums = vec![];
    let mut lengths = vec![];
    let mut total_lengths = vec![];
    let mut times = vec![];
    let mut total_length = 0;
    //benchmark!("making pairs",
    for i in 0..path.steps.len() {
        let (from, weight, to) = &path.steps[i];
        total_length += weight;
        let mut sum = 0;
        for j in i..path.steps.len() {
            let (from2, weight2, to2) = &path.steps[j];
            sum += weight2;
            //if sum <= max_time as u32 {
                pairs.push((*from, *to2));
                sums.push(sum);
                lengths.push(j - i + 1);
                total_lengths.push(total_length);
                times.push(i as u64);
            //}
        }
    }//);
    
    let paths = pairs.iter().zip(sums.iter()).zip(lengths.iter()).zip(total_lengths.iter()).zip(times.iter()).collect::<Vec<_>>();
    /*for i in 0..paths.len() {
        let (((((from, to), sum), length), total_length), time) = paths[i];
        println!("DEBUG {} --{}-> {} at start time {}", from, sum, to, time);
    }*/

    let mut subpaths = vec![];
    for (((((from, to), sum), length), total_length), time) in paths {
        subpaths.push(SubPath {
            from: *from,
            to: *to,
            time_start: *time as u64,
            total_length: if *sum <= max_time as u32 { *sum } else { u32::MAX },
            invalidated_at: U256::zero(),
        });
    }

    let tree = get_correlated_paths_tree(path, max_time);

    //println!("Tree : {:?}", tree.fields);

    //println!("{:?}", subpaths);
    
    return subpaths;
}


// Perfect balanced Binary Tree
/*
Node 0 : sons = 1 2
Node 1 : sons = 3 4
Node 2 : sons = 4 5
*/
struct MergedPathTree {
    fields : Vec<SubPath>
}

impl MergedPathTree {

    fn get_prof(idx : usize) -> usize {
        //stack exchange get inverse of formula for sum of intergers from 1 to n?
        return (((2 * idx) as f64 + 0.25).sqrt() - 0.5) as usize + 1;
    }

    fn get_idx_left_son(idx : usize) -> usize {
        return idx + Self::get_prof(idx);
    }

    fn get_idx_right_son(idx : usize) -> usize {
        return Self::get_idx_left_son(idx) + 1;
    }

    fn get_idx_parents(idx : usize) -> (Option<usize>, Option<usize>) {
        if idx == 0 {
            return (None, None);
        }
        let prof = Self::get_prof(idx);
        /*if prof > idx {
            println!("prof is {}, idx is {}", prof, idx);
            panic!("prof is greater than idx");
        }*/
        let (minr, maxr) = Self::get_range(prof);
        let left_parent = if idx == minr {
            None
        } else {
            Some(idx - (prof-1))
        };
        let right_parent = if idx == maxr {
            None
        } else {
            Some(idx - (prof-1) + 1)
        };
        return (left_parent, right_parent);
    }

    fn get_range(prof : usize) -> (usize, usize) {
        if prof == 0 {
            return (0,0)
        }
        let truc = ((prof + 1) * prof) / 2;
        return (truc - prof, truc - 1);
    }
}

fn get_correlated_paths_tree(path : &Path, max_time: u64) -> MergedPathTree {
    if path.steps.is_empty() {
        return MergedPathTree {
            fields : vec![]
        }
    }
    let nb_elems = (path.steps.len() * (path.steps.len() + 1)) / 2;
    let mut elems = vec![ SubPath {
        from : 0,
        to : 0,
        time_start : 0,
        total_length : 0,
        invalidated_at : U256::zero()
    } ; nb_elems];
    //println!("{:?}", path);
    let leaf_index = ((path.steps.len() - 1) * (path.steps.len()))/ 2;
    //println!("Leaf start {}", leaf_index);
    for (i, step) in path.steps.iter().enumerate() {
        //println!("elems[{}] = {:?}", leaf_index + i, step);
        elems[leaf_index + i] = SubPath {
            from : step.0,
            to : step.2,
            time_start : 0,
            total_length : step.1,
            invalidated_at : U256::zero()
        };
    }
    for p in (0..MergedPathTree::get_prof(leaf_index)-1).rev() {
        let (minr, maxr) = MergedPathTree::get_range(p+1);
        for i in minr..=maxr {
            let il = MergedPathTree::get_idx_left_son(i);
            let ir = MergedPathTree::get_idx_right_son(i);

            let shared = MergedPathTree::get_idx_left_son(ir);
            let shared_len = if shared < nb_elems {
                elems[shared].total_length
            } else {
                0
            };
            let left = &elems[il];
            let right = &elems[ir];
            //println!("left {} + right {} - shared {} with len {}", left.total_length, right.total_length, shared, shared_len);
            let new_path = SubPath {
                from : left.from,
                to : right.to,
                time_start : 0,
                // TODO : fix that because they get doubled up
                total_length : (left.total_length + right.total_length) - shared_len,
                invalidated_at : U256::zero()
            };
            //println!("elems[{i}] = {new_path:?} = {il} + {ir}");
            elems[i] = new_path;
        }
    }

    //println!("Done {:?}", elems);
    
    return MergedPathTree {
        fields : elems
    }
}


// FIXME : EWWWWWWWWWWWWW
fn invalidate_path_tree_layer(path: &mut MergedPathTree, layer: usize, deleted_edges: &DeletedLinksMatrix, max_time: u64, dst_mat_del: &mut Vec<DistanceMatrix>) {
    // look at all the leaves, if they are invalidated, then invalidate the parent
    let (minr, maxr) = MergedPathTree::get_range(layer);
    for i in minr..=maxr {
        let invalidated_at_matrix = deleted_edges.links[path.fields[i].from as usize][path.fields[i].to as usize];
        path.fields[i].invalidated_at |= invalidated_at_matrix;
        let (leaf_invalidated, leaf_length, (left_parent, right_parent)) = {
            let leaf = &path.fields[i];
            (leaf.invalidated_at, leaf.total_length, MergedPathTree::get_idx_parents(i))
        };
        let (leaf_from, leaf_to) = (path.fields[i].from, path.fields[i].to);
        //println!("Leaf from {} to {} with bitmask {:?}", leaf_from, leaf_to, leaf_invalidated);
        if let Some(left_parent) = left_parent {
            let left_invalidated = if (leaf_length as usize) < 256 {
                path.fields[left_parent].invalidated_at | leaf_invalidated >> leaf_length as usize | leaf_invalidated
            } else {
                path.fields[left_parent].invalidated_at | leaf_invalidated
            };
            path.fields[left_parent].invalidated_at = left_invalidated;
        }
        if let Some(right_parent) = right_parent {
            let right_invalidated = if (leaf_length as usize) < 256 {
                path.fields[right_parent].invalidated_at | leaf_invalidated >> leaf_length as usize | leaf_invalidated
            } else {
                path.fields[right_parent].invalidated_at | leaf_invalidated
            };
            path.fields[right_parent].invalidated_at = right_invalidated;
        }

        // write 0 in the distance matrix
        for t in 0..max_time {
            // if they're deleted, write 0 in the distance matrix
            if leaf_invalidated & (U256::one() << t) != U256::zero() {
                // check for u32::MAX because we don't want to write 0 in the distance matrix if the path is already infinite
                if dst_mat_del[t as usize][leaf_from as usize][leaf_to as usize] != u32::MAX {
                    dst_mat_del[t as usize][leaf_from as usize][leaf_to as usize] = 0;
                }
                //dst_mat_del[t as usize][leaf_from as usize][leaf_to as usize] = 0;
            }
        }
    }
    if layer == 0 {
        return;
    }
    invalidate_path_tree_layer(path, layer - 1, deleted_edges, max_time, dst_mat_del);
}

fn invalidate_path_tree(paths: &mut Vec<MergedPathTree>, deleted_edges: &DeletedLinksMatrix, max_time: u64, dst_mat_del: &mut Vec<DistanceMatrix>) {
    for path in paths {
        let max_prof = MergedPathTree::get_prof(path.fields.len() - 1);
        invalidate_path_tree_layer(path, max_prof, deleted_edges, max_time, dst_mat_del);
    }
}

fn johnson(graph: &Graph, max_time: u64) -> (Vec<CorrelatedPaths>, DistanceMatrix) {
    let mut dst_mat = vec![vec![u32::MAX; graph.max_node_index as usize]; graph.max_node_index as usize];
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                dst_mat[i as usize][j as usize] = 1;
            }
        }
    }
    let mut paths = vec![];
    for (node, _) in &graph.nodes {
        for (node2, _) in &graph.nodes {
            if node == node2 {
                continue;
            }
            let path = dijkstra(graph, *node, *node2, max_time, &mut dst_mat);
            let subpaths = old_get_correlated_paths(&path, max_time, &dst_mat);
            paths.push(subpaths);
        }
    }
    return (paths, dst_mat);
}

// Todo : j'ai l'impression que c'est moins rapide tf?
fn new_johnson(graph: &Graph, max_time: u64) -> (DistanceMatrix, Vec<CorrelatedPathsTwo>) {
    let mut dst_mat = vec![vec![0; graph.max_node_index as usize]; graph.max_node_index as usize];
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                dst_mat[i as usize][j as usize] = 1;
            }
        }
    }
    let mut paths = vec![];
    for (node, _) in &graph.nodes {
        let mut nb_dijkstras_done = 0;
        for (node2, _) in graph.nodes.iter() {
            if node == node2 {
                continue;
            }
            if dst_mat[*node as usize][*node2 as usize] != 0 {
                continue;
            }
            nb_dijkstras_done += 1;
            let path = dijkstra(graph, *node, *node2, max_time, &mut dst_mat);
            if path.steps.is_empty() {
                dst_mat[*node as usize][*node2 as usize] = u32::MAX;
                continue;
            }
            /*dst_mat[*node as usize][*node2 as usize] = {
                let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
                if sum <= max_time as u32 {
                    sum
                } else {
                    u32::MAX
                }
            };*/
            /*if (path.from == 1 && path.to == 30) {
                println!("{:?}", path);
                let subpaths = get_correlated_paths(&path, max_time, &dst_mat);
                println!("{:?}", subpaths);
            }*/
            let subpaths = get_correlated_paths(&path, max_time);
            for subpath in &subpaths {
                let (from, to, time_start, total_length) = (subpath.from, subpath.to, subpath.time_start, subpath.total_length);
                /*if (path.from == 1 && path.to == 30) {
                    println!("Subpath from {} to {} at time {} with total length {}", from, to, time_start, total_length);
                }*/
                if dst_mat[from as usize][to as usize] == 0 {
                    dst_mat[from as usize][to as usize] = total_length;
                } else {
                    // FIXME : en soit est ce qu'on a besoin de faire min ici? Si c'est le cas, alors on a un bug
                    if total_length != dst_mat[from as usize][to as usize] {
                        println!("Subpath from {} to {} at time {} with total length {}", from, to, time_start, total_length);
                        println!("Old value : {}", dst_mat[from as usize][to as usize]);
                        println!("New value : {}", total_length);
                    }
                    dst_mat[from as usize][to as usize] = std::cmp::min(dst_mat[from as usize][to as usize], total_length);
                }
            }
            paths.push(CorrelatedPathsTwo {
                main_path: path,
                subpaths,
            });
        }
        // TODO : si path pas trouvé, alors mettre a infini
        println!("Dijkstras done for node {} : {}", node, nb_dijkstras_done);
        /*if (*node < 5) {
            print_matrix(&dst_mat);
        }*/
    }

    return (dst_mat, paths);
        //print_matrix(&dst_mat);
    //}
    //return dst_mat;
}

// Todo : j'ai l'impression que c'est moins rapide tf?
fn newer_johnson(graph: &Graph, max_time: u64) -> (DistanceMatrix, Vec<MergedPathTree>) {
    let mut dst_mat = vec![vec![0; graph.max_node_index as usize]; graph.max_node_index as usize];
    for i in 0..graph.max_node_index {
        for j in 0..graph.max_node_index {
            if i == j {
                dst_mat[i as usize][j as usize] = 1;
            }
        }
    }
    let mut paths = vec![];
    for (node, _) in &graph.nodes {
        let mut nb_dijkstras_done = 0;
        for (node2, _) in graph.nodes.iter() {
            if node == node2 {
                continue;
            }
            if dst_mat[*node as usize][*node2 as usize] != 0 {
                continue;
            }
            nb_dijkstras_done += 1;
            let path = dijkstra(graph, *node, *node2, max_time, &mut dst_mat);
            if path.steps.is_empty() {
                dst_mat[*node as usize][*node2 as usize] = u32::MAX;
                continue;
            }
            /*dst_mat[*node as usize][*node2 as usize] = {
                let sum = path.steps.iter().map(|(_, w, _)| w).sum::<u32>();
                if sum <= max_time as u32 {
                    sum
                } else {
                    u32::MAX
                }
            };*/
            /*if (path.from == 1 && path.to == 30) {
                println!("{:?}", path);
                let subpaths = get_correlated_paths(&path, max_time, &dst_mat);
                println!("{:?}", subpaths);
            }*/
            let subpaths = get_correlated_paths_tree(&path, max_time);
            for subpath in &subpaths.fields {
                let (from, to, time_start, total_length) = (subpath.from, subpath.to, subpath.time_start, subpath.total_length);
                /*if (path.from == 1 && path.to == 30) {
                    println!("Subpath from {} to {} at time {} with total length {}", from, to, time_start, total_length);
                }*/
                if (total_length > max_time as u32) {
                    dst_mat[from as usize][to as usize] = u32::MAX;
                    continue;
                }
                if dst_mat[from as usize][to as usize] == 0 {
                    dst_mat[from as usize][to as usize] = total_length;
                } else {
                    // FIXME : en soit est ce qu'on a besoin de faire min ici? Si c'est le cas, alors on a un bug
                    if total_length != dst_mat[from as usize][to as usize] {
                        println!("Subpath from {} to {} at time {} with total length {}", from, to, time_start, total_length);
                        println!("Old value : {}", dst_mat[from as usize][to as usize]);
                        println!("New value : {}", total_length);
                    }
                    dst_mat[from as usize][to as usize] = std::cmp::min(dst_mat[from as usize][to as usize], total_length);
                }
            }
            paths.push(subpaths);
        }
        // TODO : si path pas trouvé, alors mettre a infini
        println!("Dijkstras done for node {} : {}", node, nb_dijkstras_done);
        /*if (*node < 5) {
            print_matrix(&dst_mat);
        }*/
    }
    return (dst_mat, paths);
        //print_matrix(&dst_mat);
    //}
    //return dst_mat;
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimeVaryingGraph {
    max_time: u64,
    edges: Vec<Edge>,
    deleted_edges: Vec<(u64, u64, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimeVaryingEdge {
    from: u64,
    to: u64,
    weight: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    let mut annex_edges = vec![];
    for edge in &graph.edges {
        let mut weights = vec![];
        for t in 0..max_time {
            // TODO : rewrite that to be more efficient and less ugly
            let mut wait = 0;
            for deleted_edge in deleted_edges {
                if deleted_edge.from == edge.from && deleted_edge.to == edge.to {
                    while deleted_edge.times.contains(&(t + wait)) {
                        wait += 1;
                    }
                }
            }
            let weight_t = get_weight2(edge, t) + wait as u32;
            // Remove weights that are too high
            weights.push(if weight_t + t as u32 <= max_time as u32 { weight_t } else { u32::MAX });
        }
        annex_edges.push(TimeVaryingEdge {
            from: edge.from,
            to: edge.to,
            weight: weights,
        });
    }
    return annex_edges;
}



// Unvalidates all shortest paths that go through a deleted edge
// By writing 0 in the distance matrix (0 means it has to be recomputed)
// Infinity means that there is no path, but not that it has to be recomputed
fn invalidate_deleted_edges(paths: &Vec<CorrelatedPaths>, deleted_edges: &DeletedLinksMatrix, max_time: u64, dst_mat_del: &mut Vec<DistanceMatrix>) {
    for path in paths {
        let main_path = &path.main_path;
        let mut sum = 0;
        for i in 0..main_path.steps.len() {
            let (from, weight, to) = &main_path.steps[i];
            for t in 0..max_time {
                if deleted_edges.links[*from as usize][*to as usize] & (U256::one() << t) != U256::zero() {
                    // invalidate the distance matrix
                    let subpaths = &path.subpaths;
                    for t in 0..max_time {
                        for subpath in subpaths {
                            dst_mat_del[t as usize][subpath.index_start][subpath.index_end] = 0;
                        }
                    }
                    break;
                }
            }
        }
    }
}

fn invalidate_deleted_edges_new(paths: &Vec<CorrelatedPathsTwo>, deleted_edges: &DeletedLinksMatrix, max_time: u64, dst_mat_del: &mut Vec<DistanceMatrix>) {
    for path in paths {
        let main_path = &path.main_path;
        let mut sum = 0;
        for i in 0..main_path.steps.len() {
            let (from, weight, to) = &main_path.steps[i];
            for t in 0..max_time {
                if deleted_edges.links[*from as usize][*to as usize] & (U256::one() << t) != U256::zero() {
                    // invalidate the distance matrix
                    let subpaths = &path.subpaths;
                    //for t in 0..max_time {
                        for subpath in subpaths {
                            if subpath.total_length == u32::MAX {
                                continue;
                            }
                            dst_mat_del[t as usize][subpath.from as usize][subpath.to as usize] = 0;
                        }
                    //}
                    break;
                }
            }
        }
    }
}

fn graph_to_temporal(graph: &Graph, max_time: u64, deleted_edges: &Vec<DeletedLink>) -> AnnexTimeVaryingGraph {
    let mut edges : Vec<TimeVaryingEdge> = vec![];
    let (mut dst_mat_undel, mut paths) = benchmark!("newer johnson", newer_johnson(graph, max_time));
    //let (paths,_) = benchmark!("johnson", johnson(graph, max_time));
    //let mut dst_mat_undel = benchmark!("from_corr_paths", from_shortest_corr_paths(&paths, &graph.nodes, graph.max_node_index, max_time, deleted_edges));
    print_matrix(&dst_mat_undel);
    // TODO : change that to infinity once debuggin is done
    let mut dst_mat_del = vec![dst_mat_undel.clone(); max_time as usize];

    let annex_edges = benchmark!("create_phantom_edges", create_phantom_edges(&graph, max_time, &deleted_edges, &mut dst_mat_del));
    
    // TODO : yeah this is very space consuming, probably better to switch to a sparse matrix or hashmap
    let deleted_edges_matrix = benchmark!("deleted_edges_matrix", {
        let max_node_index = graph.nodes.iter().map(|(n, _)| n).max().unwrap() + 1;
        let mut deleted_edges_matrix = vec![vec![U256::zero(); max_node_index as usize]; max_node_index as usize];
        for deleted_edge in deleted_edges {
            for t in &deleted_edge.times {
                deleted_edges_matrix[deleted_edge.from as usize][deleted_edge.to as usize] |= U256::one() << *t;
            }
        }
        let deleted_edges_matrix = DeletedLinksMatrix {
            links: deleted_edges_matrix,
        };
        deleted_edges_matrix
    });

    println!("Deleted edges matrix : {:?}", deleted_edges_matrix.links);

    benchmark!("invalidate_deleted_edges", invalidate_path_tree(&mut paths, &deleted_edges_matrix, max_time, &mut dst_mat_del));
    println!("Annex edges : {:?}", annex_edges);

    return AnnexTimeVaryingGraph {
        max_time,
        nodes: graph.nodes.clone(),
        edges: annex_edges,
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

struct TemporalPath {
    path : Path,
    start_time : u64,
}

// TODO : optimize to write every min distance computed
fn temporal_dijkstra(graph: &AnnexTimeVaryingGraph, start: Node, end: Node, max_time: u64, time: u64, max_node_index: u64, distance_matrix: &DistanceMatrix) -> Path {
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
            continue;
        }
        visited[node as usize] = true;
        for edge in &graph.edges {
            if edge.from == node {
                if dist[node as usize] == u32::MAX {
                    continue;
                }
                //println!("dist[node as usize] = {}, edge.weight[time as usize] = {}", dist[node as usize], edge.weight[time as usize]);
                if edge.weight[time as usize] == u32::MAX {
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
    return Path {
        from: start,
        to: end,
        steps,
    };
}

fn recompute_all_distance_matrix(graph: &mut AnnexTimeVaryingGraph) {
    let max_node_index = graph.nodes.iter().map(|(n, _)| n).max().unwrap() + 1;
    for t in 0..graph.max_time {
        for i in 0..graph.dst_mat_undel.len() {
            for j in 0..graph.dst_mat_undel[i].len() {
                if graph.dst_mat_del[t as usize][i][j] == 0 {
                    //println!("Recomputing distance from {} to {} at time {}", i, j, t);
                    // TODO : take adventage of the fact that we already have the shortest path from a bunch of other nodes
                    let path = temporal_dijkstra(graph, i as u64, j as u64, graph.max_time, t, max_node_index, &graph.dst_mat_del[t as usize]);
                    if path.steps.is_empty() {
                        graph.dst_mat_del[t as usize][i][j] = u32::MAX;
                        continue;
                    }
                    let subpaths = get_correlated_paths(&path, graph.max_time);
                    for subpath in &subpaths {
                        let (from, to, time_start, total_length) = (subpath.from, subpath.to, subpath.time_start, subpath.total_length);
                        let new_time = t + time_start;
                        if new_time < graph.max_time {
                            graph.dst_mat_del[new_time as usize][from as usize][to as usize] = total_length;
                        }
                    }
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
    
    let nb_nodes = 200;
    let edges : Vec<Edge> = (0..nb_nodes).map(|i| {
        Edge {
            from: i,
            to: i + 1,
            weight: 1,
        }
    }).collect();
    //let deleted_edges = vec![(0, 1, 0), (0, 1, 1), (5, 6, 0), (7, 8, 14), (8, 9, 14), (9, 10, 8), (10, 11, 2), (11, 12, 0), (22, 23, 0), (55, 56, 12), (56, 57, 12), (57, 58, 12), (58, 59, 12), (59, 60, 12), (60, 61, 12), (61, 62, 12), (62, 63, 12), (63, 64, 12), (64, 65, 12), (65, 66, 12), (66, 67, 12), (67, 68, 12), (68, 69, 45), (69, 70, 12), (70, 71, 73), (71, 72, 112), (72, 73, 12), (73, 74, 12), (74, 75, 12)];
    let deleted_edges = vec![DeletedLink {
        from: 0,
        to: 1,
        times: vec![0, 1],
    }, DeletedLink {
        from: 5,
        to: 6,
        times: vec![0],
    }, DeletedLink {
        from: 7,
        to: 8,
        times: vec![14],
    }, DeletedLink {
        from: 8,
        to: 9,
        times: vec![14],
    }, DeletedLink {
        from: 9,
        to: 10,
        times: vec![8],
    }, DeletedLink {
        from: 10,
        to: 11,
        times: vec![2],
    }, DeletedLink {
        from: 11,
        to: 12,
        times: vec![0],
    }, DeletedLink {
        from: 22,
        to: 23,
        times: vec![0],
    }, DeletedLink {
        from: 55,
        to: 56,
        times: vec![12],
    }, DeletedLink {
        from: 56,
        to: 57,
        times: vec![12],
    }, DeletedLink {
        from: 57,
        to: 58,
        times: vec![12],
    }, DeletedLink {
        from: 58,
        to: 59,
        times: vec![12],
    }, DeletedLink {
        from: 59,
        to: 60,
        times: vec![12],
    }, DeletedLink {
        from: 60,
        to: 61,
        times: vec![12],
    }, DeletedLink {
        from: 61,
        to: 62,
        times: vec![12],
    }, DeletedLink {
        from: 62,
        to: 63,
        times: vec![12],
    }, DeletedLink {
        from: 63,
        to: 64,
        times: vec![12],
    }, DeletedLink {
        from: 64,
        to: 65,
        times: vec![12],
    }, DeletedLink {
        from: 65,
        to: 66,
        times: vec![12],
    }, DeletedLink {
        from: 66,
        to: 67,
        times: vec![12],
    }, DeletedLink {
        from: 67,
        to: 68,
        times: vec![12],
    }, DeletedLink {
        from: 68,
        to: 69,
        times: vec![45],
    }, DeletedLink {
        from: 69,
        to: 70,
        times: vec![12],
    }, DeletedLink {
        from: 70,
        to: 71,
        times: vec![73],
    }, DeletedLink {
        from: 71,
        to: 72,
        times: vec![112],
    }, DeletedLink {
        from: 72,
        to: 73,
        times: vec![12],
    }, DeletedLink {
        from: 73,
        to: 74,
        times: vec![12],
    }, DeletedLink {
        from: 74,
        to: 75,
        times: vec![12],
    }];
    let nodes = (0..nb_nodes+1).map(|i| (i, vec![(i + 1, 1)])).collect();
    let max_time = 50;

    /*let nb_nodes = 10;
    let edges : Vec<Edge> = (0..nb_nodes).map(|i| {
        Edge {
            from: i,
            to: i + 1,
            weight: 1,
        }
    }).collect();
    let deleted_edges = vec![DeletedLink {
        from: 1,
        to: 2,
        times: vec![0, 1],
    }];
    let nodes = (0..nb_nodes+1).map(|i| (i, vec![(i + 1, 1)])).collect();
    let max_time = 5;*/

    /*let edges = vec![
            Edge {
                from: 0,
                to: 1,
                weight: 2,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 1,
            },
            Edge {
                from: 2,
                to: 1,
                weight: 2,
            },
        ];
    let deleted_edges = vec![DeletedLink {
        from: 0,
        to: 1,
        times: vec![0, 1],
    }];
    let nodes = vec![(0, vec![(1, 1), (2, 1)]), (1, vec![]), (2, vec![(1, 1)])];
    let max_time = 6;*/


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

    let mut annex_graph= graph_to_temporal(&graph, max_time, &deleted_edges);
    //print_graph(&graph);
    // print_timegraph(&time_graph);
    println!("Graph with invalidation");
    print_annex_graph(&annex_graph);

    benchmark!("recompute_all_distance_matrix", recompute_all_distance_matrix(&mut annex_graph));
    let (sum, reachables) = sum_dma(&annex_graph.dst_mat_del, 20);
    println!("Sum of 1/distance is : {}", sum);
    println!("Reachables : {}", reachables);
    print_annex_graph(&annex_graph);

    /*let merged = MergedPathTree {
        fields : Vec::new()
    };

    for i in 0..15 {
        println!("{} | {} : {} {}",MergedPathTree::get_prof(i), i, MergedPathTree::get_idx_left_son(i), MergedPathTree::get_idx_right_son(i));
    }

    let path = Path {
        from : 0,
        to : 5,
        steps : vec![(0,1,1), (1,1,2), (2,1,3), (3,1,4), (4,1,5)]
    };
    print_path(&path);
    let tree = get_correlated_paths_tree(&path, 10);
    println!("tree : {:?}",&tree.fields);*/

    /*let from = 0;
    let to = 20;
    let steps = (from..to).map(|i| (i, 1, i + 1)).collect();
    let path = Path {
        from,
        to,
        steps,
    };
    let correlated_paths = get_correlated_paths(&path, 20, &vec![]);*/
}
