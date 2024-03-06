// A project to compute distances on time-varying graphs
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

struct Edge {
    node1: u64,
    node2: u64,
    weight: u64,
}

struct TimeVaryingGraph {
    // The maximum time for the graph
    max_time: u64,
    // The graph is represented as a list of edges
    edges: Vec<Edge>,
    // Deleted edges (if [u, v, t] is in deleted_edges, then the edge (u, v) is deleted at time t)
    deleted_edges: Vec<(u64, u64, u64)>,
}

fn print_graph(graph: &TimeVaryingGraph) {
    println!("Original graph with deleted links with max time : {}", graph.max_time);
    // print time by time
    for t in 0..graph.max_time {
        println!("Time {}", t);
        for edge in &graph.edges {
            // if the edge is not deleted at time t, print it
            if !graph.deleted_edges.contains(&(edge.node1, edge.node2, t)) {
                println!("({}, {}) with weight {}", edge.node1, edge.node2, edge.weight);
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TimeVaryingEdge {
    node1: u64,
    node2: u64,
    weight: Vec<u64>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct AnnexTimeVaryingGraph {
    // The maximum time for the graph
    max_time: u64,
    // The graph is represented as a list of edges
    edges: Vec<TimeVaryingEdge>,
}

fn print_annex_graph(graph: &AnnexTimeVaryingGraph) {
    println!("New graph with max time : {}", graph.max_time);
    // print time by time
    for t in 0..graph.max_time {
        println!("Time {}", t);
        for edge in &graph.edges {
            // if the edge is not deleted at time t, print it
            println!("{} --{}--> {}",
                     edge.node1,
                     edge.weight[t as usize],
                     edge.node2); 
        }
    }
}

fn get_weight2(edge : &Edge, _time : u64) -> u64 {
    // TODO : why did i use time ?
    edge.weight
}

fn from_time_varying_graph_to_annex(graph : &TimeVaryingGraph) -> AnnexTimeVaryingGraph {
    let mut annex_edges = vec![];
    //let mut nodes_to_do_dijkstra = vec![];
    for edge in &graph.edges {
        let mut weights = vec![];
        for t in 0..graph.max_time {
            if graph.deleted_edges.contains(&(edge.node1, edge.node2, t)) {
                // get wait time since
                let mut wait_time = 1;
                while graph.deleted_edges.contains(&(edge.node1, edge.node2, t + wait_time)) {
                    wait_time += 1;
                }
                //let weight_at_t = get_weight2(edge, t);
                weights.push(get_weight2(edge, t + wait_time) + wait_time);
                //nodes_to_do_dijkstra.push((edge.node1, edge.node2, t));
            } else {
                weights.push(get_weight2(edge, t));
            }
        }
        annex_edges.push(TimeVaryingEdge {
            node1: edge.node1,
            node2: edge.node2,
            weight: weights,
        });
    }
    let annex_graph = AnnexTimeVaryingGraph {
        max_time: graph.max_time,
        edges: annex_edges,
    };

    // Since it will already do it in the johnson algorithm, we don't need to do it here
    // Do Dijkstra for each node
    /*for (node1, node2, t) in nodes_to_do_dijkstra {
        let distance = temporal_dijkstra(&annex_graph, node1, node2, t);
        println!("Distance from {} to {} at time {} is {}", node1, node2, t, distance);
        // if the distance is lower than the weight, update the weight
        println!("Edge from {} to {} at time {} has weight {}", node1, node2, t, annex_graph.edges[node1 as usize].weight[t as usize]);
        let old_weight = annex_graph.edges[node1 as usize].weight[t as usize];
        if distance < old_weight {
            annex_graph.edges[node1 as usize].weight[t as usize] = distance;
        }
    }*/

    return annex_graph;
    
}


struct DistanceMatrix {
    distances : Vec<Vec<u64>>,
}

fn get_weight(source : u64, destination : u64, time : u64, graph : &AnnexTimeVaryingGraph) -> u64 {
    //println!("Getting weight from {} to {} at time {} : {}", source, destination, time, graph.edges[source as usize].weight[time as usize]);
    if source == destination {
        return 1;
    }
    // find an edge from source to destination
    for edge in &graph.edges {
        if edge.node1 == source && edge.node2 == destination {
            return edge.weight[time as usize];
        }
    }
    return u64::MAX;
}

fn temporal_dijkstra(graph : &AnnexTimeVaryingGraph, source : u64, destination : u64, time : u64, distance_matrices : *mut Vec<Vec<Vec<u64>>>, max_time : u64) -> u64 {
    //println!("Dijkstra from {} to {} at time {}", source, destination, time);
    
    // if already computed, return the value
    unsafe {
        let distance_matrices = distance_matrices as *mut Vec<Vec<Vec<u64>>>;
        if (*distance_matrices)[time as usize][source as usize][destination as usize] != 0 {
            //println!("Already computed");
            return (*distance_matrices)[time as usize][source as usize][destination as usize];
        }
    }

    // If it's a direct edge, return the weight
    let mut distances = vec![];
    let mut visited = vec![];
    for _ in 0..graph.edges.len() {
        distances.push(u64::MAX);
        visited.push(false);
    }
    for i in 0..graph.edges.len() {
        distances[i] = get_weight(source, i as u64, time, graph);
    }
    distances[source as usize] = 1; // C'est la diagonale (est ce que ca change quelque chose ?)
    for _ in 0..graph.edges.len() {

        let mut min_distance = u64::MAX;
        let mut min_index = 0;
        for i in 0..graph.edges.len() {
            if !visited[i] && distances[i] <= min_distance {
                min_distance = distances[i];
                min_index = i;
            }
        }
        visited[min_index] = true;
        for i in 0..graph.edges.len() {
            let weight = get_weight(min_index as u64, i as u64, time, graph);
            if !visited[i] && distances[min_index] != u64::MAX && weight != u64::MAX && distances[min_index] + weight < distances[i] {
                distances[i] = distances[min_index] + weight;
            }
        }
    }
    //print all the other shortest paths found from source
    for i in 0..distances.len() {
        /*if visited[i] {
            println!("Found min dist from {} to {} is {} at time {}", source, i, distances[i], time);
        }*/
        if visited[i] {
            //println!("Found min dist from {} to {} is {} at time {}", source, i, distances[i], time);
            unsafe {
                let distance_matrices = distance_matrices as *mut Vec<Vec<Vec<u64>>>;
                (*distance_matrices)[time as usize][source as usize][i as usize] = distances[i];
            }
        }
    }
    return distances[destination as usize];
}

fn full_parallel_johnson(graph : &AnnexTimeVaryingGraph) -> Vec<DistanceMatrix> {
    let mut distance_matrices = vec![vec![vec![0; graph.edges.len()]; graph.edges.len()]; graph.max_time as usize];
    let ptr_distance_matrices = &mut distance_matrices as *mut Vec<Vec<Vec<u64>>>;
    let ptr_distance_matrices_int = ptr_distance_matrices as usize;
    (0..graph.max_time).into_par_iter().for_each(|t| {
        (0..graph.edges.len()).into_par_iter().for_each(|i| {
            (0..graph.edges.len()).into_par_iter().for_each(|j| {
                
                // we can skip the computation if i == j, but if we do it it won't change the result, so we don't care
                if i == j {
                    return;
                }
                let distance_matrices = ptr_distance_matrices_int as *mut Vec<Vec<Vec<u64>>>;
                let distance = temporal_dijkstra(graph, i as u64, j as u64, t, distance_matrices, graph.max_time);
                unsafe {
                    (*distance_matrices)[t as usize][i as usize][j as usize] = distance;
                }

                
                let distance_matrices = ptr_distance_matrices_int as *mut Vec<Vec<Vec<u64>>>;
                let distance_matrices = unsafe { &mut *distance_matrices };
                if distance_matrices[t as usize][i as usize][j as usize] == u64::MAX {
                    return;
                }
                // Infinity if we took too much time to reach the destination
                if t + distance_matrices[t as usize][i as usize][j as usize] > graph.max_time {
                    distance_matrices[t as usize][i as usize][j as usize] = u64::MAX;
                }
            })
        })
    });
    let mut distance_matrices = unsafe { std::mem::transmute::<Vec<Vec<Vec<u64>>>, Vec<DistanceMatrix>>(distance_matrices) };
    // for each element, if time + elem > max_time, turn it to infinity
    /*for t in 0..distance_matrices.len() as u64 {
        for i in 0..distance_matrices[t as usize].distances.len() {
            for j in 0..distance_matrices[t as usize].distances.len() {
                if distance_matrices[t as usize].distances[i as usize][j as usize] == u64::MAX {
                    continue;
                }
                // Infinity if we took too much time to reach the destination
                if t + distance_matrices[t as usize].distances[i as usize][j as usize] > graph.max_time {
                    distance_matrices[t as usize].distances[i as usize][j as usize] = u64::MAX;
                }
            }
        }
    }*/
    distance_matrices
}

fn time_parallel_johnson(graph : &AnnexTimeVaryingGraph) -> Vec<DistanceMatrix> {
    // only parallelize the time
    let mut distance_matrices = vec![vec![vec![0; graph.edges.len()]; graph.edges.len()]; graph.max_time as usize];
    let ptr_distance_matrices = &mut distance_matrices as *mut Vec<Vec<Vec<u64>>>;
    let ptr_distance_matrices_int = ptr_distance_matrices as usize;
    (0..graph.max_time).into_par_iter().for_each(|t| {
        for i in 0..graph.edges.len() {
            for j in 0..graph.edges.len() {
                if i == j {
                    continue;
                }
                let distance_matrices = ptr_distance_matrices_int as *mut Vec<Vec<Vec<u64>>>;
                let distance_matrices = unsafe { &mut *distance_matrices };
                let distance = temporal_dijkstra(graph, i as u64, j as u64, t, distance_matrices, graph.max_time);
                unsafe {
                    (*distance_matrices)[t as usize][i as usize][j as usize] = distance;
                }
                if distance != u64::MAX && distance != 0 {
                    if t + distance > graph.max_time {
                        distance_matrices[t as usize][i as usize][j as usize] = u64::MAX;
                    } else {
                        distance_matrices[t as usize][i as usize][j as usize] = distance;
                    }
                }
            }
        }
    });
    let mut distance_matrices = unsafe { std::mem::transmute::<Vec<Vec<Vec<u64>>>, Vec<DistanceMatrix>>(distance_matrices) };
    distance_matrices

}

fn johnson(graph : &AnnexTimeVaryingGraph) -> Vec<DistanceMatrix> {
    let mut distance_matrices = vec![vec![vec![0; graph.edges.len()]; graph.edges.len()]; graph.max_time as usize];
    for t in 0..graph.max_time {
        // for i in 0..graph.edges.len() {
        //     for j in 0..graph.edges.len() {
        //         if i == j {
        //             distance_matrices[t as usize][i as usize][j as usize] = 1;
        //         } else {
        //             distance_matrices[t as usize][i as usize][j as usize] = u64::MAX;
        //         }
        //     }
        // }
        for i in 0..graph.edges.len() {
            for j in 0..graph.edges.len() {
                if i == j {
                     distance_matrices[t as usize][i as usize][j as usize] = 1;
                }
                else {
                    let distance = temporal_dijkstra(graph, i as u64, j as u64, t, &mut distance_matrices as *mut Vec<Vec<Vec<u64>>>, graph.max_time);
                    if distance != u64::MAX {
                        if t + distance > graph.max_time {
                            distance_matrices[t as usize][i as usize][j as usize] = u64::MAX;
                        } else {
                            distance_matrices[t as usize][i as usize][j as usize] = distance;
                        }
                    }
                }
            }
        }
        println!("Time {}", t);
    }
    let mut distance_matrices = unsafe { std::mem::transmute::<Vec<Vec<Vec<u64>>>, Vec<DistanceMatrix>>(distance_matrices) };
    distance_matrices
}

/*fn parallel_johnson(graph : &AnnexTimeVaryingGraph) -> Vec<DistanceMatrix> {
    let result = (0..graph.max_time).into_par_iter().map(|t| {
        let mut distances = vec![];
        for i in 0..graph.edges.len() {
            let mut row = vec![];
            for j in 0..graph.edges.len() {
                if i == j {
                    row.push(1);
                } else {
                    row.push(u64::MAX);
                }
            }
            distances.push(row);
        }
        for i in 0..graph.edges.len() {
            for j in 0..graph.edges.len() {
                if i == j {
                    continue;
                }
                let distance = temporal_dijkstra(graph, i as u64, j as u64, t);
                distances[i as usize][j as usize] = distance;
            }
        }
        DistanceMatrix {
            distances: distances,
        }
    }).collect();
    result
}

fn gpu_parallel_johnson(graph : &AnnexTimeVaryingGraph) -> (f64, u64) {
    let mut distance_matrices = vec![vec![vec![0; graph.edges.len()]; graph.edges.len()]; graph.max_time as usize];
    let distance_matrices_ptr = unsafe { std::mem::transmute::<&mut Vec<Vec<Vec<u64>>>, *mut Vec<Vec<Vec<u64>>>>(&mut distance_matrices) };
    let distance_matrices_ptr_int = distance_matrices_ptr as usize;
    let sum = Arc::new(Mutex::new(0.0));
    let reachables = AtomicU64::new(0);
    (0..graph.max_time).into_par_iter().for_each(|t| {
        (0..graph.edges.len()).into_par_iter().for_each(|i| {
            (0..graph.edges.len()).into_par_iter().for_each(|j| {
                if i == j {
                    return;
                }
                let distance = temporal_dijkstra(graph, i as u64, j as u64, t);
                unsafe {
                    let distance_matrices = distance_matrices_ptr_int as *mut Vec<Vec<Vec<u64>>>;
                    (*distance_matrices)[t as usize][i as usize][j as usize] = distance;
                }
                if distance != u64::MAX && distance != 0 {
                    reachables.fetch_add(1, Ordering::Relaxed);
                    let mut sum = sum.lock().unwrap();
                    *sum += 1.0 / (distance as f64);
                }
            })
        })
    });
    let sum = sum.lock().unwrap();
    let mut sum = *sum;
    sum -= graph.edges.len() as f64;
    sum /= graph.max_time as f64 * (graph.edges.len() * (graph.edges.len() - 1)) as f64;
    let reachables = reachables.load(Ordering::Relaxed);
    return (sum, reachables);
}*/

fn print_distance_matrix(distance_matrices : &Vec<DistanceMatrix>) {
    for t in 0..distance_matrices.len() {
        println!("Time {}", t);
        for i in 0..distance_matrices[t as usize].distances.len() {
            for j in 0..distance_matrices[t as usize].distances.len() {
                let distance = distance_matrices[t as usize].distances[i as usize][j as usize];
                if distance == u64::MAX {
                    print!("∞ ");
                } else {
                    print!("{} ", distance);
                }
            }
            println!();
        }
    }
}

fn pretty_print(distance_matrices : &Vec<DistanceMatrix>) {
    // For each time, tell me all the distances
    for t in 0..distance_matrices.len() {
        println!("Time {}", t);
        for i in 0..distance_matrices[t as usize].distances.len() {
            for j in 0..distance_matrices[t as usize].distances.len() {
                if i == j {
                    continue;
                }
                let distance = distance_matrices[t as usize].distances[i as usize][j as usize];
                if distance == u64::MAX {
                    println!("Distance from {} to {} is infinite", i, j);
                } else {
                    println!("Distance from {} to {} is {}", i, j, distance);
                }
            }
        }
    }
}

fn sum_dma(dma : &Vec<DistanceMatrix>) -> (f64, u64) {
    let mut sum = 0.0;
    let mut reachables = 0;
    for t in 0..dma.len() {
        for i in 0..dma[t as usize].distances.len() {
            for j in 0..dma[t as usize].distances.len() {
                if i == j {
                    continue;
                }
                let distance = dma[t as usize].distances[i as usize][j as usize];
                if distance != u64::MAX && distance != 0 {
                    sum += 1.0 / (distance as f64);
                    reachables += 1;
                }
            }
        }
    }
    sum -= dma[0 as usize].distances.len() as f64;
    sum /= dma.len() as f64 * (dma[0 as usize].distances.len() * (dma[0 as usize].distances.len() - 1)) as f64;
    (sum, reachables)
}
fn main() {
    let graph = TimeVaryingGraph {
        max_time: 20,
        /*edges: vec![
            Edge {
                node1: 0,
                node2: 1,
                weight: 3,
            },
            Edge {
                node1: 0,
                node2: 2,
                weight: 1,
            },
            Edge {
                node1: 2,
                node2: 1,
                weight: 1,
            },
            Edge {
                node1: 5,
                node2: 1,
                weight: 1,
            },
            Edge {
                node1: 5,
                node2: 2,
                weight: 1,
            },
            Edge {
                node1: 5,
                node2: 3,
                weight: 1,
            },
            Edge {
                node1: 3,
                node2: 4,
                weight: 1,
            },
            Edge {
                node1: 4,
                node2: 5,
                weight: 1,
            },
        ],*/
        edges : (0..200).map(|i| {
            Edge {
                node1: i,
                node2: i + 1,
                weight: 1,
            }
        }).collect(),
        deleted_edges: vec![(0, 1, 0), (0, 1, 1)]
    };
    //print_graph(&graph);
    let annex_graph = from_time_varying_graph_to_annex(&graph);
    //print_annex_graph(&annex_graph);
    
    /*let distance_matrices = johnson(&annex_graph);
    println!("Single thread");
    //print_distance_matrix(&distance_matrices);
    let (sum, reachables) = sum_dma(&distance_matrices);
    println!("Robustness is {} with {} reachables", sum, reachables);*/

    /*let distance_matrices = full_parallel_johnson(&annex_graph);
    println!("GPU parallel");
    //print_distance_matrix(&distance_matrices);
    let (sum, reachables) = sum_dma(&distance_matrices);
    println!("Robustness is {} with {} reachables", sum, reachables);

    let distance_matrices = time_parallel_johnson(&annex_graph);
    println!("Parallel");
    //print_distance_matrix(&distance_matrices);
    let (sum, reachables) = sum_dma(&distance_matrices);
    println!("Robustness is {} with {} reachables", sum, reachables);*/

    // benchmark the two algorithms
    let start = std::time::Instant::now();
    let distance_matrices = johnson(&annex_graph);
    let duration = start.elapsed();
    println!("Single thread");
    //print_distance_matrix(&distance_matrices);
    let (sum, reachables) = sum_dma(&distance_matrices);
    println!("Robustness is {} with {} reachables", sum, reachables);
    println!("Time elapsed in single thread is: {:?}", duration);

    let start = std::time::Instant::now();
    let distance_matrices = full_parallel_johnson(&annex_graph);
    let duration = start.elapsed();
    println!("GPU parallel");
    //print_distance_matrix(&distance_matrices);
    let (sum, reachables) = sum_dma(&distance_matrices);
    println!("Robustness is {} with {} reachables", sum, reachables);
    println!("Time elapsed in GPU parallel is: {:?}", duration);

    let start = std::time::Instant::now();
    let distance_matrices = time_parallel_johnson(&annex_graph);
    let duration = start.elapsed();
    println!("Parallel");
    //print_distance_matrix(&distance_matrices);
    let (sum, reachables) = sum_dma(&distance_matrices);
    println!("Robustness is {} with {} reachables", sum, reachables);
    println!("Time elapsed in parallel is: {:?}", duration);

}