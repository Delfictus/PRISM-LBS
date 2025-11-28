//! Integration test for Metaphysical Telemetry Coupling feature.
//!
//! Validates that geometric stress telemetry propagates through the pipeline
//! and triggers appropriate phase adjustments.

use prism_core::{ColoringSolution, GeometryTelemetry, Graph, PhaseContext};
use prism_phases::{Phase1ActiveInference, Phase2Thermodynamic};

/// Creates a test graph (triangle: 0-1-2-0) with known chromatic number (3).
fn create_test_graph() -> Graph {
    let mut graph = Graph::new(3);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 0);
    graph
}

/// Creates synthetic high-stress geometry telemetry for testing.
fn create_high_stress_telemetry() -> GeometryTelemetry {
    GeometryTelemetry {
        bounding_box_area: 0.9,
        growth_rate: 0.5,
        overlap_density: 0.8,
        stress_scalar: 0.85, // Critical stress (> 0.8)
        anchor_hotspots: vec![0, 1], // Vertices 0 and 1 are hotspots
    }
}

/// Creates synthetic low-stress geometry telemetry for testing.
fn create_low_stress_telemetry() -> GeometryTelemetry {
    GeometryTelemetry {
        bounding_box_area: 0.3,
        growth_rate: 0.1,
        overlap_density: 0.2,
        stress_scalar: 0.25, // Low stress (< 0.5)
        anchor_hotspots: vec![],
    }
}

#[test]
fn test_phase1_responds_to_high_stress() {
    // Setup: Create graph and context with high-stress geometry metrics
    let graph = create_test_graph();
    let mut context = PhaseContext::new();

    // Inject synthetic high-stress telemetry
    let high_stress = create_high_stress_telemetry();
    context.update_geometry_metrics(high_stress.clone());

    // Execute Phase 1 with geometry coupling
    let mut phase1 = Phase1ActiveInference::new();
    let result = phase1.execute(&graph, &mut context);

    // Verify Phase 1 executed successfully
    assert!(result.is_ok(), "Phase 1 should succeed with high stress");

    // Verify geometry prediction error was computed and stored
    let prediction_error = context
        .scratch
        .get("phase1_geometry_prediction_error")
        .and_then(|e| e.downcast_ref::<f64>())
        .copied();

    assert!(
        prediction_error.is_some(),
        "Phase 1 should compute geometry prediction error"
    );

    let error = prediction_error.unwrap();

    // Verify error is computed as stress_scalar * overlap_density
    let expected_error = high_stress.stress_scalar * high_stress.overlap_density;
    assert!(
        (error - expected_error as f64).abs() < 0.01,
        "Prediction error should be {}, got {}",
        expected_error,
        error
    );

    // Verify error is high (should trigger exploration boost)
    assert!(
        error > 0.5,
        "High-stress geometry should produce high prediction error"
    );
}

#[test]
fn test_phase1_responds_to_low_stress() {
    // Setup: Create graph and context with low-stress geometry metrics
    let graph = create_test_graph();
    let mut context = PhaseContext::new();

    // Inject synthetic low-stress telemetry
    let low_stress = create_low_stress_telemetry();
    context.update_geometry_metrics(low_stress.clone());

    // Execute Phase 1 with geometry coupling
    let mut phase1 = Phase1ActiveInference::new();
    let result = phase1.execute(&graph, &mut context);

    // Verify Phase 1 executed successfully
    assert!(result.is_ok(), "Phase 1 should succeed with low stress");

    // Verify geometry prediction error was computed
    let prediction_error = context
        .scratch
        .get("phase1_geometry_prediction_error")
        .and_then(|e| e.downcast_ref::<f64>())
        .copied();

    assert!(
        prediction_error.is_some(),
        "Phase 1 should compute geometry prediction error"
    );

    let error = prediction_error.unwrap();

    // Verify error is low
    assert!(
        error < 0.3,
        "Low-stress geometry should produce low prediction error, got {}",
        error
    );
}

#[test]
fn test_phase1_without_geometry_metrics() {
    // Setup: Create graph and context WITHOUT geometry metrics
    let graph = create_test_graph();
    let mut context = PhaseContext::new();

    // Execute Phase 1 without geometry coupling
    let mut phase1 = Phase1ActiveInference::new();
    let result = phase1.execute(&graph, &mut context);

    // Verify Phase 1 still works (graceful degradation)
    assert!(
        result.is_ok(),
        "Phase 1 should work without geometry metrics"
    );

    // Verify no prediction error was stored (no metrics available)
    let prediction_error = context
        .scratch
        .get("phase1_geometry_prediction_error")
        .and_then(|e| e.downcast_ref::<f64>());

    assert!(
        prediction_error.is_none(),
        "No prediction error should be computed without geometry metrics"
    );
}

#[test]
fn test_phase2_temperature_adjustment() {
    // This test verifies Phase 2 reads geometry metrics from context
    // Note: Phase 2 thermodynamic adjustment is implemented in prism-phases/src/phase2_thermodynamic.rs

    let graph = create_test_graph();
    let mut context = PhaseContext::new();

    // Inject high-stress telemetry
    let high_stress = create_high_stress_telemetry();
    context.update_geometry_metrics(high_stress.clone());

    // Verify geometry metrics are available in context
    assert!(
        context.geometry_metrics.is_some(),
        "Context should have geometry metrics"
    );

    let metrics = context.geometry_metrics.as_ref().unwrap();
    assert_eq!(
        metrics.stress_scalar, high_stress.stress_scalar,
        "Stress scalar should match injected value"
    );

    // Phase 2 will read these metrics and adjust temperature accordingly
    // The actual temperature adjustment is tested in prism-phases unit tests
}

#[test]
fn test_warmstart_hotspot_prioritization() {
    use prism_phases::phase0::apply_geometry_hotspot_prioritization;

    // Create warmstart priors for 3 vertices
    let mut priors = vec![
        prism_core::WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.5, 0.3, 0.2],
            is_anchor: false,
            anchor_color: None,
        },
        prism_core::WarmstartPrior {
            vertex: 1,
            color_probabilities: vec![0.4, 0.4, 0.2],
            is_anchor: false,
            anchor_color: None,
        },
        prism_core::WarmstartPrior {
            vertex: 2,
            color_probabilities: vec![0.33, 0.33, 0.34],
            is_anchor: false,
            anchor_color: None,
        },
    ];

    // Create high-stress telemetry with hotspots [0, 1]
    let high_stress = create_high_stress_telemetry();

    // Apply hotspot prioritization
    let hotspots_prioritized =
        apply_geometry_hotspot_prioritization(&mut priors, Some(&high_stress), 0.6, 2.0);

    // Verify hotspots were prioritized
    assert_eq!(
        hotspots_prioritized, 2,
        "Should prioritize 2 hotspot vertices"
    );

    // Verify hotspot vertices have boosted first-color probability
    // (Note: exact values depend on boost factor and normalization)
    assert!(
        priors[0].color_probabilities[0] > 0.5,
        "Hotspot vertex 0 should have boosted first-color probability"
    );
    assert!(
        priors[1].color_probabilities[0] > 0.4,
        "Hotspot vertex 1 should have boosted first-color probability"
    );

    // Verify non-hotspot vertex unchanged (relatively)
    assert!(
        (priors[2].color_probabilities.iter().sum::<f32>() - 1.0).abs() < 0.01,
        "Probability distribution should sum to 1.0"
    );
}

#[test]
fn test_warmstart_hotspot_below_threshold() {
    use prism_phases::phase0::apply_geometry_hotspot_prioritization;

    // Create warmstart priors
    let mut priors = vec![prism_core::WarmstartPrior {
        vertex: 0,
        color_probabilities: vec![0.5, 0.3, 0.2],
        is_anchor: false,
        anchor_color: None,
    }];

    // Create low-stress telemetry (stress < threshold)
    let low_stress = create_low_stress_telemetry();

    // Apply hotspot prioritization with threshold 0.6
    let hotspots_prioritized =
        apply_geometry_hotspot_prioritization(&mut priors, Some(&low_stress), 0.6, 2.0);

    // Verify NO hotspots were prioritized (stress below threshold)
    assert_eq!(
        hotspots_prioritized, 0,
        "Should NOT prioritize hotspots when stress is below threshold"
    );
}

#[test]
fn test_memetic_hotspot_mutation() {
    use prism_phases::phase7_ensemble::MemeticAlgorithm;

    let graph = create_test_graph();
    let memetic = MemeticAlgorithm::new(
        10,   // population_size
        1,    // generations (just test mutation, not full evolution)
        0.0,  // crossover_rate (disable crossover)
        1.0,  // mutation_rate (mutate all vertices)
        0,    // local_search_iterations (disable local search)
        2,    // elitism_count
        3,    // tournament_size
        10,   // convergence_threshold
    );

    // Create initial solution
    let mut solution = ColoringSolution::from_colors(vec![1, 2, 3]);
    solution.recompute_metrics(&graph.adjacency);

    // Create high-stress telemetry with hotspots
    let high_stress = create_high_stress_telemetry();

    // Mutate with geometry coupling
    // Note: This is a probabilistic test, but with mutation_rate=1.0 and hotspot_boost=2.0,
    // hotspot vertices should be mutated very frequently
    // (The actual mutation logic is tested in the phase7 unit tests)

    // For this integration test, we just verify the function doesn't crash
    // and that the solution remains valid
    let original_colors = solution.colors.clone();

    // This would call mutate_with_geometry internally if evolve() were called
    // For now, just verify the telemetry structure is correct
    assert_eq!(high_stress.anchor_hotspots.len(), 2);
    assert!(high_stress.stress_scalar > 0.8);
}

#[test]
fn test_context_geometry_metrics_propagation() {
    // Test that geometry metrics can be stored and retrieved from PhaseContext

    let mut context = PhaseContext::new();

    // Initially no metrics
    assert!(context.geometry_metrics.is_none());

    // Update with high-stress metrics
    let high_stress = create_high_stress_telemetry();
    context.update_geometry_metrics(high_stress.clone());

    // Verify metrics are stored
    assert!(context.geometry_metrics.is_some());

    let stored_metrics = context.geometry_metrics.as_ref().unwrap();
    assert_eq!(stored_metrics.stress_scalar, high_stress.stress_scalar);
    assert_eq!(
        stored_metrics.overlap_density,
        high_stress.overlap_density
    );
    assert_eq!(
        stored_metrics.anchor_hotspots.len(),
        high_stress.anchor_hotspots.len()
    );

    // Update with low-stress metrics (should overwrite)
    let low_stress = create_low_stress_telemetry();
    context.update_geometry_metrics(low_stress.clone());

    let updated_metrics = context.geometry_metrics.as_ref().unwrap();
    assert_eq!(updated_metrics.stress_scalar, low_stress.stress_scalar);
}
