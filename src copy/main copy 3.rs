struct Edge {
    from: u64,
    to: u64,
    weight: u32,
}

struct TimeVaryingGraph {
    // The maximum time for the graph
    max_time: u64,
    // The graph is represented as a list of edges
    edges: Vec<Edge>,
    // Deleted edges (if [u, v, t] is in deleted_edges, then the edge (u, v) is deleted at time t)
    deleted_edges: Vec<(u64, u64, u64)>,
}

type DistanceMatrix = Vec<Vec<u32>>;

struct AnnexTimeVaryingGraph {
    max_time: u64,
    dst_mat_undel: DistanceMatrix,
}

fn print_graph(graph: &TimeVaryingGraph) {
    println!("Original graph with deleted links with max time : {}", graph.max_time);
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

struct GraphAlgorithm {
    graph: AnnexTimeVaryingGraph,
    nodes_to_compute: Vec<(u64, u64, u64)>,
}

struct Path {
    from: u64,
    to: u64,
    steps: Vec<(u64, u64)>,
}

// Compute the shortest paths from every node to every other node
fn compute_shortest_paths(graph: &TimeVaryingGraph) -> AnnexTimeVaryingGraph {
    let mut dst_mat_undel = vec![vec![u32::MAX; graph.edges.len()]; graph.edges.len()];
    for edge in &graph.edges {
        dst_mat_undel[edge.from as usize][edge.to as usize] = edge.weight;
    }
    AnnexTimeVaryingGraph {
        max_time: graph.max_time,
        dst_mat_undel,
    }
}

fn print_annex_graph(graph: &AnnexTimeVaryingGraph) {
    println!("Annex graph with max time : {}", graph.max_time);
    // print time by time
    for t in 0..graph.max_time {
        println!("Time {}", t);
        for i in 0..graph.dst_mat_undel.len() {
            for j in 0..graph.dst_mat_undel[i].len() {
                println!("{} --{}--> {} : {}", i, graph.dst_mat_undel[i][j], j, t);
            }
        }
    }
}

fn main() {
    let graph = TimeVaryingGraph {
        max_time: 3,
        edges: vec![
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
        ],
        deleted_edges: vec![(0, 1, 0), (0, 1, 1)]
    };
    print_graph(&graph);
}