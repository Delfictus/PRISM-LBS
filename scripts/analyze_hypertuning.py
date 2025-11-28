#!/usr/bin/env python3
"""
PRISM Hypertuning Results Analyzer
Analyzes campaign results to identify optimal parameters for achieving 48 colors
"""

import json
import sys
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re

def load_telemetry(file_path):
    """Load and parse JSONL telemetry file"""
    results = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        if results:
            # Get the final state
            last_entry = results[-1]
            return {
                'file': os.path.basename(file_path),
                'colors': last_entry.get('final_state', {}).get('colors', 999),
                'conflicts': last_entry.get('final_state', {}).get('conflicts', 999),
                'iterations': last_entry.get('iteration', 0),
                'runtime': last_entry.get('elapsed_time', 0),
                'guard_triggers': last_entry.get('guard_triggers', 0),
                'geometric_stress': last_entry.get('metrics', {}).get('geometric_stress', 0),
                'diversity': last_entry.get('metrics', {}).get('diversity', 0)
            }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

    return None

def extract_params_from_filename(filename):
    """Extract parameters from filename like sweep_mu0.75_t3.0_a0.94_c10.0.jsonl"""
    params = {}

    # Extract mu (chemical potential)
    mu_match = re.search(r'mu([0-9.]+)', filename)
    if mu_match:
        params['mu'] = float(mu_match.group(1))

    # Extract temperature
    t_match = re.search(r'_t([0-9.]+)', filename)
    if t_match:
        params['temperature'] = float(t_match.group(1))

    # Extract annealing rate
    a_match = re.search(r'_a([0-9.]+)', filename)
    if a_match:
        params['annealing_rate'] = float(a_match.group(1))

    # Extract coupling strength
    c_match = re.search(r'_c([0-9.]+)', filename)
    if c_match:
        params['coupling_strength'] = float(c_match.group(1))

    return params

def analyze_campaign(campaign_dir):
    """Analyze all results in a campaign directory"""

    print(f"\n{'='*60}")
    print(f"PRISM HYPERTUNING ANALYSIS")
    print(f"Campaign: {campaign_dir}")
    print(f"{'='*60}\n")

    # Find all telemetry files
    telemetry_files = glob.glob(os.path.join(campaign_dir, "*.jsonl"))

    if not telemetry_files:
        print(f"No telemetry files found in {campaign_dir}")
        return

    # Load all results
    results = []
    for file_path in telemetry_files:
        result = load_telemetry(file_path)
        if result:
            # Add parameters from filename
            params = extract_params_from_filename(result['file'])
            result.update(params)

            # Calculate score
            result['score'] = abs(result['colors'] - 48) + result['conflicts'] * 1000
            results.append(result)

    if not results:
        print("No valid results found")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Sort by score
    df = df.sort_values('score')

    # 1. Victory Check
    victories = df[(df['colors'] == 48) & (df['conflicts'] == 0)]

    if not victories.empty:
        print("🎯 VICTORY CONFIGURATIONS FOUND!")
        print("="*40)
        for idx, row in victories.iterrows():
            print(f"\nConfiguration: {row['file']}")
            if 'mu' in row:
                print(f"  Chemical Potential μ: {row['mu']:.2f}")
            if 'temperature' in row:
                print(f"  Initial Temperature: {row['temperature']:.1f}")
            if 'annealing_rate' in row:
                print(f"  Annealing Rate: {row['annealing_rate']:.3f}")
            if 'coupling_strength' in row:
                print(f"  Coupling Strength: {row['coupling_strength']:.1f}")
            print(f"  Runtime: {row['runtime']:.2f}s")
            print(f"  Iterations: {row['iterations']}")
        print()

    # 2. Best Results
    print("\nTOP 10 RESULTS:")
    print("="*40)
    print(f"{'Colors':>7} {'Conflicts':>10} {'Score':>7} {'File':<30}")
    print("-"*40)

    for idx, row in df.head(10).iterrows():
        status = ""
        if row['colors'] == 48 and row['conflicts'] == 0:
            status = " ✅"
        elif row['conflicts'] == 0:
            status = " 🔶"

        print(f"{row['colors']:>7.0f} {row['conflicts']:>10.0f} {row['score']:>7.0f} {row['file']:<30}{status}")

    # 3. Statistical Summary
    print("\n\nSTATISTICAL SUMMARY:")
    print("="*40)
    print(f"Total attempts: {len(df)}")
    print(f"Best score: {df['score'].min():.0f}")
    print(f"Colors range: {df['colors'].min():.0f} - {df['colors'].max():.0f}")
    print(f"Conflict-free solutions: {len(df[df['conflicts'] == 0])}")
    print(f"Solutions with exactly 48 colors: {len(df[df['colors'] == 48])}")

    # 4. Parameter Analysis (if available)
    if 'mu' in df.columns and len(df) > 5:
        print("\n\nPARAMETER ANALYSIS:")
        print("="*40)

        # Find parameters that correlate with good results
        param_cols = ['mu', 'temperature', 'annealing_rate', 'coupling_strength']
        available_params = [col for col in param_cols if col in df.columns]

        if available_params:
            # Get top 20% best results
            top_20_pct = df.head(max(1, len(df) // 5))

            print("\nOptimal parameter ranges (top 20% of results):")
            for param in available_params:
                if not top_20_pct[param].isna().all():
                    mean_val = top_20_pct[param].mean()
                    std_val = top_20_pct[param].std()
                    min_val = top_20_pct[param].min()
                    max_val = top_20_pct[param].max()

                    print(f"\n{param}:")
                    print(f"  Best range: {min_val:.3f} - {max_val:.3f}")
                    print(f"  Mean ± std: {mean_val:.3f} ± {std_val:.3f}")

    # 5. Near Misses Analysis
    near_misses = df[
        ((df['colors'] >= 46) & (df['colors'] <= 50)) &
        (df['conflicts'] == 0)
    ]

    if len(near_misses) > 0:
        print("\n\nNEAR MISSES (46-50 colors, 0 conflicts):")
        print("="*40)
        print(f"Found {len(near_misses)} near misses")

        colors_46 = len(near_misses[near_misses['colors'] == 46])
        colors_47 = len(near_misses[near_misses['colors'] == 47])
        colors_49 = len(near_misses[near_misses['colors'] == 49])
        colors_50 = len(near_misses[near_misses['colors'] == 50])

        if colors_46 > 0:
            print(f"  46 colors: {colors_46} (need less compression)")
        if colors_47 > 0:
            print(f"  47 colors: {colors_47} (very close!)")
        if colors_49 > 0:
            print(f"  49 colors: {colors_49} (slightly over)")
        if colors_50 > 0:
            print(f"  50 colors: {colors_50} (need more compression)")

    # 6. Recommendations
    print("\n\nRECOMMENDATIONS:")
    print("="*40)

    if not victories.empty:
        print("✅ Victory achieved! Use the winning configuration.")
    else:
        # Analyze what adjustments are needed
        best = df.iloc[0]

        if best['conflicts'] > 0:
            print("❗ Priority: Resolve conflicts")
            print("   - Decrease annealing rate (slower cooling)")
            print("   - Increase conflict penalty")
            print("   - Increase Kempe chain attempts")

        if best['conflicts'] == 0:
            if best['colors'] < 48:
                print("📉 Need less aggressive compression:")
                print("   - Decrease chemical potential μ")
                print("   - Decrease coupling strength")
                print("   - Increase temperature")
            elif best['colors'] > 48:
                print("📈 Need more aggressive compression:")
                print("   - Increase chemical potential μ")
                print("   - Increase coupling strength")
                print("   - Increase color penalty")

    # 7. Save summary to file
    summary_file = os.path.join(campaign_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Campaign Analysis: {datetime.now().isoformat()}\n")
        f.write(f"Total attempts: {len(df)}\n")
        f.write(f"Best result: {df.iloc[0]['colors']} colors, {df.iloc[0]['conflicts']} conflicts\n")
        if not victories.empty:
            f.write(f"VICTORY ACHIEVED: {len(victories)} winning configurations\n")

    print(f"\n\nAnalysis complete. Summary saved to: {summary_file}")

    # Return best configuration for further processing
    return df.iloc[0] if not df.empty else None

def main():
    if len(sys.argv) < 2:
        # Find most recent campaign
        campaigns = glob.glob("campaigns/hypertuning_*")
        if campaigns:
            campaign_dir = max(campaigns, key=os.path.getctime)
            print(f"No campaign specified, using most recent: {campaign_dir}")
        else:
            print("Usage: python analyze_hypertuning.py <campaign_directory>")
            print("Example: python analyze_hypertuning.py campaigns/hypertuning_20240120_143022")
            sys.exit(1)
    else:
        campaign_dir = sys.argv[1]

    if not os.path.isdir(campaign_dir):
        print(f"Error: Directory {campaign_dir} does not exist")
        sys.exit(1)

    best_result = analyze_campaign(campaign_dir)

    # If we have a near-miss, suggest next steps
    if best_result and best_result['score'] > 0 and best_result['score'] < 10:
        print("\n\n" + "="*60)
        print("NEXT STEPS FOR FINE-TUNING:")
        print("="*60)

        if 'mu' in best_result:
            mu = best_result['mu']
            if best_result['colors'] < 48:
                print(f"Try μ values: {mu-0.05:.2f}, {mu-0.02:.2f}")
            elif best_result['colors'] > 48:
                print(f"Try μ values: {mu+0.02:.2f}, {mu+0.05:.2f}")

        print("\nRun focused search with:")
        print("./scripts/hypertuner_v2.sh --fine-tune")

if __name__ == "__main__":
    main()