//! Integration test for CMA-ES optimization phase

use prism_core::{CmaState, Graph, PhaseContext, PhaseController};
use prism_physics::{CmaEsConfig, CmaEsPhaseController};

/// Test that CMA-ES optimization improves a graph coloring solution
#[test]
fn test_cma_es_optimization() {
    // Initialize logging
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .is_test(true)
        .try_init();

    // Create a small test graph (K4 - complete graph with 4 vertices)
    let mut graph = Graph::new(4);
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(0, 3);
    graph.add_edge(1, 2);
    graph.add_edge(1, 3);
    graph.add_edge(2, 3);

    // Create CMA-ES configuration
    let config = CmaEsConfig {
        population_size: 20,
        initial_sigma: 0.5,
        max_iterations: 50,
        target_fitness: Some(4.0), // K4 needs 4 colors
        use_gpu: false, // CPU only for test
    };

    // Create phase controller
    let mut controller = CmaEsPhaseController::new(config);

    // Create execution context
    let mut context = PhaseContext::new();

    // Execute CMA-ES phase
    let outcome = controller.execute(&graph, &mut context).unwrap();

    // Verify successful execution
    assert!(outcome.is_success());

    // Check that CMA state was stored
    let cma_state = context.get_cma_state().expect("CMA state should be present");

    // Verify optimization occurred
    assert!(cma_state.generation > 0, "Should have run at least one generation");
    assert!(
        cma_state.best_fitness < f32::INFINITY,
        "Should have found a finite fitness value"
    );
    assert!(
        cma_state.best_solution.len() == graph.num_vertices,
        "Solution should have one parameter per vertex"
    );

    // Log results
    log::info!("CMA-ES Test Results:");
    log::info!("  Generations: {}", cma_state.generation);
    log::info!("  Best fitness: {:.3}", cma_state.best_fitness);
    log::info!("  Convergence: {:.3}", cma_state.convergence_metric);
    log::info!("  Sigma: {:.3}", cma_state.sigma);

    // Check if we found a valid coloring (best case: fitness around 4.0 for K4)
    assert!(
        cma_state.best_fitness < 100.0,
        "Fitness should be reasonable for a small graph"
    );
}

/// Test CMA-ES convergence detection
#[test]
fn test_cma_es_convergence() {
    // Create a trivial graph (single vertex)
    let graph = Graph::new(1);

    // Create CMA-ES configuration
    let config = CmaEsConfig {
        population_size: 10,
        initial_sigma: 0.1,
        max_iterations: 100,
        target_fitness: None,
        use_gpu: false,
    };

    // Create phase controller
    let mut controller = CmaEsPhaseController::new(config);

    // Create execution context
    let mut context = PhaseContext::new();

    // Execute CMA-ES phase
    controller.execute(&graph, &mut context).unwrap();

    // Check convergence
    let cma_state = context.get_cma_state().expect("CMA state should be present");

    // For a single vertex, should converge very quickly
    assert!(
        cma_state.convergence_metric > 0.0,
        "Should have some convergence for trivial problem"
    );
    assert!(
        cma_state.best_fitness < 10.0,
        "Trivial problem should have low fitness"
    );
}

/// Test CMA-ES telemetry emission
#[test]
fn test_cma_es_telemetry() {
    use std::fs;
    use std::path::Path;

    // Create test telemetry file path
    let telemetry_path = "test_cma_telemetry.jsonl";

    // Clean up any existing file
    let _ = fs::remove_file(telemetry_path);

    // Create a small test graph
    let mut graph = Graph::new(3);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);

    // Create CMA-ES configuration
    let config = CmaEsConfig {
        population_size: 10,
        initial_sigma: 0.3,
        max_iterations: 10,
        target_fitness: None,
        use_gpu: false,
    };

    // Create phase controller
    let mut controller = CmaEsPhaseController::new(config);

    // Create execution context
    let mut context = PhaseContext::new();

    // Execute and get outcome
    let outcome = controller.execute(&graph, &mut context).unwrap();

    // Extract telemetry from outcome
    if let prism_core::PhaseOutcome::Success { telemetry, .. } = outcome {
        // Create telemetry event
        use prism_pipeline::telemetry::TelemetryEvent;

        let metrics: std::collections::HashMap<String, f64> = telemetry
            .iter()
            .filter_map(|(k, v)| {
                v.as_f64().map(|f| (k.clone(), f))
            })
            .collect();

        let event = TelemetryEvent::new(
            "PhaseX-CMA",
            metrics,
            &prism_core::PhaseOutcome::success(),
        );

        // Add CMA state to event
        let event = if let Some(cma_state) = context.get_cma_state() {
            event.with_cma(cma_state)
        } else {
            event
        };

        // Write to file
        event.write_json(telemetry_path).unwrap();

        // Verify file was created
        assert!(Path::new(telemetry_path).exists());

        // Read and parse JSON
        let content = fs::read_to_string(telemetry_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();

        // Verify CMA telemetry fields
        assert!(json["cma"].is_object());
        assert!(json["cma"]["best_fitness"].is_number());
        assert!(json["cma"]["generation"].is_number());
        assert!(json["cma"]["convergence_metric"].is_number());
        assert!(json["cma"]["sigma"].is_number());

        // Clean up
        let _ = fs::remove_file(telemetry_path);
    } else {
        panic!("Expected success outcome with telemetry");
    }
}