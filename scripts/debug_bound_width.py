#!/usr/bin/env python3
"""
Diagnostic script to identify instances with largest bound width discrepancies
between luna and abcrown.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import ast

def parse_bounds(bounds_str):
    """Parse bounds string like '[-465.13739]' or '[-465.13739, 123.45]' into list of floats."""
    try:
        # Remove brackets and split by comma
        bounds_str = bounds_str.strip('[]')
        if not bounds_str:
            return []
        # Split by comma and convert to float
        bounds = [float(x.strip()) for x in bounds_str.split(',')]
        return bounds
    except Exception as e:
        print(f"Warning: Could not parse bounds '{bounds_str}': {e}")
        return []

def load_csv(filepath):
    """Load CSV file and return list of records."""
    # Increase field size limit for large bound arrays
    csv.field_size_limit(sys.maxsize)
    
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

def compute_bound_width_from_bounds(lower_str, upper_str):
    """Compute bound width from lower and upper bound strings."""
    lower_bounds = parse_bounds(lower_str)
    upper_bounds = parse_bounds(upper_str)
    
    if len(lower_bounds) != len(upper_bounds):
        return None
    
    if len(lower_bounds) == 0:
        return None
    
    # Compute mean of (upper - lower)
    widths = [u - l for l, u in zip(lower_bounds, upper_bounds)]
    return sum(widths) / len(widths)

def analyze_discrepancies(luna_file, abcrown_file):
    """Analyze bound width discrepancies between luna and abcrown."""
    
    # Load data
    print(f"Loading {luna_file}...")
    luna_records = load_csv(luna_file)
    print(f"Loaded {len(luna_records)} luna records")
    
    print(f"Loading {abcrown_file}...")
    abcrown_records = load_csv(abcrown_file)
    print(f"Loaded {len(abcrown_records)} abcrown records")
    
    # Create lookup by (benchmark, slurm_id)
    luna_dict = {}
    for r in luna_records:
        key = (r['benchmark'], r['slurm_id'])
        luna_dict[key] = r
    
    abcrown_dict = {}
    for r in abcrown_records:
        key = (r['benchmark'], r['slurm_id'])
        abcrown_dict[key] = r
    
    # Find matching instances
    discrepancies = []
    for key in luna_dict:
        if key not in abcrown_dict:
            continue
        
        luna_rec = luna_dict[key]
        abcrown_rec = abcrown_dict[key]
        
        # Skip if either has missing bound_width
        if luna_rec['bound_width'] == '--' or abcrown_rec['bound_width'] == '--':
            continue
        
        try:
            luna_width = float(luna_rec['bound_width'])
            abcrown_width = float(abcrown_rec['bound_width'])
            
            # Compute ratio (luna / abcrown)
            if abcrown_width > 0:
                ratio = luna_width / abcrown_width
            elif luna_width > 0:
                ratio = float('inf')
            else:
                ratio = 1.0
            
            # Compute absolute difference
            diff = luna_width - abcrown_width
            
            # Also compute from bounds if available
            luna_width_from_bounds = None
            if luna_rec.get('lower_bounds') and luna_rec.get('upper_bounds'):
                luna_width_from_bounds = compute_bound_width_from_bounds(
                    luna_rec['lower_bounds'], luna_rec['upper_bounds'])
            
            abcrown_width_from_bounds = None
            if abcrown_rec.get('lower_bounds') and abcrown_rec.get('upper_bounds'):
                abcrown_width_from_bounds = compute_bound_width_from_bounds(
                    abcrown_rec['lower_bounds'], abcrown_rec['upper_bounds'])
            
            discrepancies.append({
                'benchmark': key[0],
                'slurm_id': key[1],
                'luna_width': luna_width,
                'abcrown_width': abcrown_width,
                'ratio': ratio,
                'diff': diff,
                'luna_width_from_bounds': luna_width_from_bounds,
                'abcrown_width_from_bounds': abcrown_width_from_bounds,
                'luna_lower': luna_rec.get('lower_bounds', ''),
                'luna_upper': luna_rec.get('upper_bounds', ''),
                'abcrown_lower': abcrown_rec.get('lower_bounds', ''),
                'abcrown_upper': abcrown_rec.get('upper_bounds', ''),
            })
        except (ValueError, KeyError) as e:
            print(f"Warning: Skipping {key}: {e}")
            continue
    
    return discrepancies

def print_summary(discrepancies):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if not discrepancies:
        print("No matching instances found!")
        return
    
    # Sort by ratio (worst first)
    sorted_by_ratio = sorted(discrepancies, key=lambda x: x['ratio'] if x['ratio'] != float('inf') else 1e10, reverse=True)
    
    print(f"\nTotal matching instances: {len(discrepancies)}")
    print(f"\nWorst 10 instances by ratio (luna/abcrown):")
    print(f"{'Benchmark':<25} {'ID':<6} {'LUNA':<15} {'ABCROWN':<15} {'Ratio':<10} {'Diff':<15}")
    print("-" * 90)
    
    for d in sorted_by_ratio[:10]:
        ratio_str = f"{d['ratio']:.2f}" if d['ratio'] != float('inf') else "inf"
        print(f"{d['benchmark']:<25} {d['slurm_id']:<6} {d['luna_width']:<15.6f} "
              f"{d['abcrown_width']:<15.6f} {ratio_str:<10} {d['diff']:<15.6f}")
    
    # Group by benchmark
    by_benchmark = defaultdict(list)
    for d in discrepancies:
        by_benchmark[d['benchmark']].append(d)
    
    print(f"\n\nBenchmark-level statistics:")
    print(f"{'Benchmark':<30} {'Count':<8} {'Avg Ratio':<12} {'Max Ratio':<12} {'Avg Diff':<15}")
    print("-" * 90)
    
    benchmark_stats = []
    for benchmark, instances in by_benchmark.items():
        ratios = [d['ratio'] for d in instances if d['ratio'] != float('inf')]
        diffs = [d['diff'] for d in instances]
        
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            max_ratio = max(ratios)
        else:
            avg_ratio = float('inf')
            max_ratio = float('inf')
        
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        
        benchmark_stats.append({
            'benchmark': benchmark,
            'count': len(instances),
            'avg_ratio': avg_ratio,
            'max_ratio': max_ratio,
            'avg_diff': avg_diff,
        })
    
    # Sort by max ratio
    benchmark_stats.sort(key=lambda x: x['max_ratio'] if x['max_ratio'] != float('inf') else 1e10, reverse=True)
    
    for stat in benchmark_stats:
        avg_ratio_str = f"{stat['avg_ratio']:.2f}" if stat['avg_ratio'] != float('inf') else "inf"
        max_ratio_str = f"{stat['max_ratio']:.2f}" if stat['max_ratio'] != float('inf') else "inf"
        print(f"{stat['benchmark']:<30} {stat['count']:<8} {avg_ratio_str:<12} "
              f"{max_ratio_str:<12} {stat['avg_diff']:<15.6f}")

def print_detailed_instances(discrepancies, top_n=20):
    """Print detailed information for top N worst instances."""
    print("\n" + "="*80)
    print(f"DETAILED ANALYSIS: Top {top_n} Worst Instances")
    print("="*80)
    
    # Sort by ratio
    sorted_by_ratio = sorted(discrepancies, key=lambda x: x['ratio'] if x['ratio'] != float('inf') else 1e10, reverse=True)
    
    for i, d in enumerate(sorted_by_ratio[:top_n], 1):
        print(f"\n{i}. {d['benchmark']} (ID: {d['slurm_id']})")
        print(f"   LUNA width:   {d['luna_width']:.6f}")
        print(f"   ABCROWN width: {d['abcrown_width']:.6f}")
        ratio_str = f"{d['ratio']:.2f}x" if d['ratio'] != float('inf') else "inf"
        print(f"   Ratio:          {ratio_str} (LUNA is {ratio_str} wider)")
        print(f"   Difference:    {d['diff']:.6f}")
        
        if d['luna_width_from_bounds'] is not None:
            print(f"   LUNA width (from bounds): {d['luna_width_from_bounds']:.6f}")
        if d['abcrown_width_from_bounds'] is not None:
            print(f"   ABCROWN width (from bounds): {d['abcrown_width_from_bounds']:.6f}")
        
        print(f"   LUNA bounds:   lower={d['luna_lower']}, upper={d['luna_upper']}")
        print(f"   ABCROWN bounds: lower={d['abcrown_lower']}, upper={d['abcrown_upper']}")

def export_worst_instances(discrepancies, output_file, top_n=50):
    """Export worst instances to CSV for further analysis."""
    sorted_by_ratio = sorted(discrepancies, key=lambda x: x['ratio'] if x['ratio'] != float('inf') else 1e10, reverse=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'benchmark', 'slurm_id', 'luna_width', 'abcrown_width', 'ratio', 'diff',
            'luna_lower', 'luna_upper', 'abcrown_lower', 'abcrown_upper'
        ])
        writer.writeheader()
        
        for d in sorted_by_ratio[:top_n]:
            writer.writerow({
                'benchmark': d['benchmark'],
                'slurm_id': d['slurm_id'],
                'luna_width': d['luna_width'],
                'abcrown_width': d['abcrown_width'],
                'ratio': d['ratio'] if d['ratio'] != float('inf') else 'inf',
                'diff': d['diff'],
                'luna_lower': d['luna_lower'],
                'luna_upper': d['luna_upper'],
                'abcrown_lower': d['abcrown_lower'],
                'abcrown_upper': d['abcrown_upper'],
            })
    
    print(f"\nExported top {top_n} worst instances to {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_bound_width.py <luna_instances.csv> <abcrown_instances.csv> [output.csv]")
        sys.exit(1)
    
    luna_file = Path(sys.argv[1])
    abcrown_file = Path(sys.argv[2])
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not luna_file.exists():
        print(f"Error: {luna_file} not found")
        sys.exit(1)
    
    if not abcrown_file.exists():
        print(f"Error: {abcrown_file} not found")
        sys.exit(1)
    
    # Analyze discrepancies
    discrepancies = analyze_discrepancies(luna_file, abcrown_file)
    
    # Print summary
    print_summary(discrepancies)
    
    # Print detailed instances
    print_detailed_instances(discrepancies, top_n=20)
    
    # Export if requested
    if output_file:
        export_worst_instances(discrepancies, output_file, top_n=50)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
