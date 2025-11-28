//! Demonstration of CMA-ES optimization for graph coloring

use prism_core::{Graph, PhaseContext, PhaseController};
use prism_physics::{CmaEsConfig, CmaEsPhaseController};

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== CMA-ES Graph Coloring Optimization Demo ===\n");

    // Create a test graph (Petersen graph - 10 vertices, 15 edges)
    let mut graph = Graph::new(10);

    // Outer pentagon
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 3);
    graph.add_edge(3, 4);
    graph.add_edge(4, 0);

    // Inner pentagram
    graph.add_edge(5, 7);
    graph.add_edge(7, 9);
    graph.add_edge(9, 6);
    graph.add_edge(6, 8);
    graph.add_edge(8, 5);

    // Spokes
    graph.add_edge(0, 5);
    graph.add_edge(1, 6);
    graph.add_edge(2, 7);
    graph.add_edge(3, 8);
    graph.add_edge(4, 9);

    println!("Graph: Petersen graph");
    println!("Vertices: {}", graph.num_vertices);
    println!("Edges: {}", graph.num_edges);
    println!("Known chromatic number: 3\n");

    // Configure CMA-ES
    let config = CmaEsConfig {
        population_size: 30,
        initial_sigma: 0.5,
        max_iterations: 100,
        target_fitness: Some(3.0), // Petersen graph needs 3 colors
        use_gpu: false,
    };

    println!("CMA-ES Configuration:");
    println!("  Population size: {}", config.population_size);
    println!("  Initial sigma: {}", config.initial_sigma);
    println!("  Max iterations: {}", config.max_iterations);
    println!("  Target fitness: {:?}\n", config.target_fitness);

    // Create phase controller
    let mut controller = CmaEsPhaseController::new(config);

    // Create execution context
    let mut context = PhaseContext::new();

    println!("Starting CMA-ES optimization...\n");

    // Execute CMA-ES phase
    let outcome = controller.execute(&graph, &mut context)?;

    // Extract results
    if let prism_core::PhaseOutcome::Success { message, telemetry } = &outcome {
        println!("✓ Optimization completed: {}\n", message);

        // Get CMA state
        if let Some(cma_state) = context.get_cma_state() {
            println!("=== CMA-ES Results ===");
            println!("Generations run: {}", cma_state.generation);
            println!("Best fitness: {:.6}", cma_state.best_fitness);
            println!("Mean fitness: {:.6}", cma_state.mean_fitness);
            println!("Fitness std dev: {:.6}", cma_state.fitness_std);
            println!("Final sigma: {:.6}", cma_state.sigma);
            println!("Convergence metric: {:.4}", cma_state.convergence_metric);
            println!("Condition number: {:.2e}", cma_state.covariance_condition);

            // Analyze solution quality
            let expected_fitness = 3.0; // Petersen graph chromatic number
            let fitness_ratio = cma_state.best_fitness / expected_fitness;

            println!("\n=== Solution Quality ===");
            if fitness_ratio < 1.5 {
                println!("✓ EXCELLENT: Found near-optimal solution!");
            } else if fitness_ratio < 2.0 {
                println!("✓ GOOD: Found reasonable solution");
            } else if fitness_ratio < 3.0 {
                println!("○ MODERATE: Solution needs improvement");
            } else {
                println!("✗ POOR: Far from optimal");
            }

            println!("Fitness ratio: {:.2}x optimal", fitness_ratio);
        }

        // Show telemetry metrics
        println!("\n=== Telemetry Metrics ===");
        for (key, value) in telemetry {
            if let Some(num) = value.as_f64() {
                println!("  {}: {:.4}", key, num);
            }
        }
    } else {
        println!("✗ Optimization failed!");
    }

    // Check if we have an improved coloring solution
    if let Some(solution) = context.best_solution.as_ref() {
        println!("\n=== Graph Coloring Solution ===");
        println!("Colors used: {}", solution.chromatic_number);
        println!("Conflicts: {}", solution.conflicts);

        if solution.conflicts == 0 {
            println!("✓ Valid coloring found!");
        } else {
            println!("✗ Solution has conflicts");
        }
    }

    Ok(())
}