#!/usr/bin/env python3
"""
Temporary script to run LIRPA with both CROWN and alpha-CROWN methods
on one instance from each benchmark in resources/regular_benchmarks.
"""

import sys
import csv
import time
import gzip
import shutil
import tempfile
import subprocess
import re
from pathlib import Path

def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

def decompress_if_needed(filepath, temp_dir):
    """
    Decompress a .gz file if needed and return path to uncompressed file.
    
    Args:
        filepath: Path object to the file (may be .gz)
        temp_dir: Path to temporary directory for decompressed files
    
    Returns:
        Path to uncompressed file
    """
    if filepath.suffix == '.gz':
        # Create temp file with same name minus .gz
        uncompressed_name = filepath.stem
        # Create a unique temp file path
        temp_file = Path(temp_dir) / f"{filepath.parent.name}_{uncompressed_name}"
        
        if not temp_file.exists():
            with gzip.open(filepath, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        return temp_file
    return filepath

def parse_lirpa_output(output):
    """
    Parse LIRPA output to extract bounds and verification result.
    
    Returns:
        dict with 'lower', 'upper', 'result', 'verified' values
    """
    info = {
        'lower': None,
        'upper': None,
        'result': None,
        'verified': None
    }
    
    lines = output.split('\n')
    
    for line in lines:
        # Look for verification result
        if 'UNSAT' in line or 'unsat' in line.lower():
            info['result'] = 'UNSAT'
            info['verified'] = True
        elif 'SAT' in line and 'UNSAT' not in line:
            info['result'] = 'SAT'
            info['verified'] = False
        elif 'UNKNOWN' in line or 'unknown' in line.lower():
            info['result'] = 'UNKNOWN'
        
        # Look for bounds
        if 'lower' in line.lower() and 'bound' in line.lower():
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if match:
                try:
                    info['lower'] = float(match.group())
                except:
                    pass
        
        if 'upper' in line.lower() and 'bound' in line.lower():
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if match:
                try:
                    info['upper'] = float(match.group())
                except:
                    pass
    
    return info

def extract_error_message(stderr, stdout):
    """Extract the most relevant error message from output."""
    combined = stderr + "\n" + stdout
    lines = combined.split('\n')
    
    # Look for specific error patterns
    error_lines = []
    for line in lines:
        if 'Error:' in line or 'error:' in line:
            error_lines.append(line.strip())
        elif 'exception' in line.lower():
            error_lines.append(line.strip())
        elif 'failed' in line.lower() and 'Error' not in line:
            error_lines.append(line.strip())
        elif 'terminating' in line.lower():
            error_lines.append(line.strip())
    
    if error_lines:
        # Return the most informative error (usually the first or longest)
        return max(error_lines, key=len) if len(error_lines) > 1 else error_lines[0]
    
    # If no specific error found, return last non-empty line of stderr
    stderr_lines = [l.strip() for l in stderr.split('\n') if l.strip()]
    if stderr_lines:
        return stderr_lines[-1]
    
    return "Unknown error"

def run_benchmark_instance(lirpa_binary, benchmark_name, onnx_path, vnnlib_path, timeout=300):
    """
    Run both CROWN and alpha-CROWN on a single benchmark instance.
    
    Returns:
        dict with results or error information
    """
    result = {
        'benchmark': benchmark_name,
        'onnx': onnx_path.name,
        'vnnlib': vnnlib_path.name,
        'crown_status': 'FAILED',
        'crown_time': 0,
        'crown_output': '',
        'crown_error': '',
        'alpha_status': 'FAILED',
        'alpha_time': 0,
        'alpha_output': '',
        'alpha_error': '',
        'error': None
    }
    
    # Run CROWN
    print(f"  Running CROWN...")
    crown_cmd = [
        str(lirpa_binary),
        str(onnx_path),
        str(vnnlib_path),
        '--method', 'crown',
        '--timeout', str(timeout)
    ]
    
    start = time.time()
    try:
        crown_proc = subprocess.run(
            crown_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        crown_time = time.time() - start
        result['crown_time'] = crown_time
        result['crown_output'] = crown_proc.stdout
        result['crown_error'] = crown_proc.stderr
        
        if crown_proc.returncode == 0:
            result['crown_status'] = 'SUCCESS'
            # Parse output for additional info
            info = parse_lirpa_output(crown_proc.stdout)
            result['crown_info'] = info
            result_str = f" ({info['result']})" if info['result'] else ""
            print(f"    ✓ CROWN completed in {format_time(crown_time)}{result_str}")
        else:
            result['crown_status'] = f'EXIT_{crown_proc.returncode}'
            error_msg = extract_error_message(crown_proc.stderr, crown_proc.stdout)
            result['crown_error_msg'] = error_msg
            print(f"    ✗ CROWN failed with exit code {crown_proc.returncode}")
            print(f"       {error_msg[:150]}")
                
    except subprocess.TimeoutExpired:
        crown_time = time.time() - start
        result['crown_status'] = 'TIMEOUT'
        result['crown_time'] = crown_time
        print(f"    ⏱ CROWN timeout after {format_time(crown_time)}")
    except Exception as e:
        result['crown_status'] = f'ERROR: {str(e)}'
        print(f"    ✗ CROWN error: {e}")
    
    # Run alpha-CROWN
    print(f"  Running alpha-CROWN...")
    alpha_cmd = [
        str(lirpa_binary),
        str(onnx_path),
        str(vnnlib_path),
        '--method', 'alpha-crown',
        '--timeout', str(timeout),
        '--iterations', '20'
    ]
    
    start = time.time()
    try:
        alpha_proc = subprocess.run(
            alpha_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        alpha_time = time.time() - start
        result['alpha_time'] = alpha_time
        result['alpha_output'] = alpha_proc.stdout
        result['alpha_error'] = alpha_proc.stderr
        
        if alpha_proc.returncode == 0:
            result['alpha_status'] = 'SUCCESS'
            # Parse output for additional info
            info = parse_lirpa_output(alpha_proc.stdout)
            result['alpha_info'] = info
            result_str = f" ({info['result']})" if info['result'] else ""
            print(f"    ✓ alpha-CROWN completed in {format_time(alpha_time)}{result_str}")
        else:
            result['alpha_status'] = f'EXIT_{alpha_proc.returncode}'
            error_msg = extract_error_message(alpha_proc.stderr, alpha_proc.stdout)
            result['alpha_error_msg'] = error_msg
            print(f"    ✗ alpha-CROWN failed with exit code {alpha_proc.returncode}")
            print(f"       {error_msg[:150]}")
                
    except subprocess.TimeoutExpired:
        alpha_time = time.time() - start
        result['alpha_status'] = 'TIMEOUT'
        result['alpha_time'] = alpha_time
        print(f"    ⏱ alpha-CROWN timeout after {format_time(alpha_time)}")
    except Exception as e:
        result['alpha_status'] = f'ERROR: {str(e)}'
        print(f"    ✗ alpha-CROWN error: {e}")
    
    return result

def main():
    print("="*80)
    print("LIRPA Benchmark Comparison: CROWN vs alpha-CROWN")
    print("="*80)
    
    # Find lirpa binary
    repo_root = Path(__file__).parent
    lirpa_binary = repo_root / "build" / "bin" / "lirpa"
    
    if not lirpa_binary.exists():
        print(f"Error: LIRPA binary not found: {lirpa_binary}")
        print("Please build LIRPA first.")
        return 1
    
    print(f"\nUsing LIRPA binary: {lirpa_binary}")
    
    # Find benchmark directories
    benchmarks_dir = repo_root / "resources" / "regular_benchmarks" / "benchmarks"
    
    if not benchmarks_dir.exists():
        print(f"Error: Benchmarks directory not found: {benchmarks_dir}")
        return 1
    
    # Get all benchmark directories
    benchmark_dirs = sorted([d for d in benchmarks_dir.iterdir() if d.is_dir()])
    print(f"Found {len(benchmark_dirs)} benchmarks")
    
    # Create temporary directory for decompressed files
    temp_dir = tempfile.mkdtemp(prefix='lirpa_benchmarks_')
    print(f"Using temporary directory: {temp_dir}\n")
    
    results = []
    timeout_per_instance = 60  # 60 seconds per method
    
    for bench_dir in benchmark_dirs:
        benchmark_name = bench_dir.name
        instances_csv = bench_dir / "instances.csv"
        
        if not instances_csv.exists():
            print(f"⊘ {benchmark_name}: No instances.csv found, skipping")
            continue
        
        print(f"\n{'─'*80}")
        print(f"Benchmark: {benchmark_name}")
        print(f"{'─'*80}")
        
        # Read first instance from CSV
        try:
            with open(instances_csv, 'r') as f:
                reader = csv.reader(f)
                first_row = next(reader)
                
                # Parse instance (format: onnx_path, vnnlib_path, timeout)
                onnx_rel_path = first_row[0]
                vnnlib_rel_path = first_row[1]
                
                onnx_path = bench_dir / onnx_rel_path
                vnnlib_path = bench_dir / vnnlib_rel_path
                
                # Check for .gz versions if the file doesn't exist
                if not onnx_path.exists():
                    gz_path = Path(str(onnx_path) + '.gz')
                    if gz_path.exists():
                        onnx_path = gz_path
                    else:
                        print(f"  ✗ ONNX file not found: {onnx_path} (or .gz version)")
                        results.append({
                            'benchmark': benchmark_name,
                            'crown_status': 'SKIPPED',
                            'alpha_status': 'SKIPPED',
                            'error': 'ONNX file not found'
                        })
                        continue
                
                if not vnnlib_path.exists():
                    gz_path = Path(str(vnnlib_path) + '.gz')
                    if gz_path.exists():
                        vnnlib_path = gz_path
                    else:
                        print(f"  ✗ VNN-LIB file not found: {vnnlib_path} (or .gz version)")
                        results.append({
                            'benchmark': benchmark_name,
                            'crown_status': 'SKIPPED',
                            'alpha_status': 'SKIPPED',
                            'error': 'VNN-LIB file not found'
                        })
                        continue
                
                # Decompress files if needed
                try:
                    print(f"  Preparing files...")
                    onnx_path = decompress_if_needed(onnx_path, temp_dir)
                    vnnlib_path = decompress_if_needed(vnnlib_path, temp_dir)
                    print(f"    ONNX: {onnx_path.name}")
                    print(f"    VNN-LIB: {vnnlib_path.name}")
                except Exception as e:
                    print(f"  ✗ Error decompressing files: {e}")
                    results.append({
                        'benchmark': benchmark_name,
                        'crown_status': 'ERROR',
                        'alpha_status': 'ERROR',
                        'error': f'Decompression error: {str(e)}'
                    })
                    continue
                
                # Run the benchmark
                result = run_benchmark_instance(
                    lirpa_binary, 
                    benchmark_name, 
                    onnx_path, 
                    vnnlib_path,
                    timeout=timeout_per_instance
                )
                results.append(result)
                
        except StopIteration:
            print(f"  ✗ No instances in CSV")
            results.append({
                'benchmark': benchmark_name,
                'crown_status': 'SKIPPED',
                'alpha_status': 'SKIPPED',
                'error': 'Empty instances.csv'
            })
        except Exception as e:
            print(f"  ✗ Error reading instances.csv: {e}")
            results.append({
                'benchmark': benchmark_name,
                'crown_status': 'ERROR',
                'alpha_status': 'ERROR',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(f"\n{'Benchmark':<25} {'CROWN':<20} {'Time':<12} {'alpha-CROWN':<20} {'Time':<12}")
    print("─"*90)
    
    crown_success = 0
    alpha_success = 0
    crown_timeout = 0
    alpha_timeout = 0
    total_crown_time = 0
    total_alpha_time = 0
    
    for r in results:
        crown_status = r.get('crown_status', 'N/A')
        alpha_status = r.get('alpha_status', 'N/A')
        crown_time = r.get('crown_time', 0)
        alpha_time = r.get('alpha_time', 0)
        
        # Format status
        if crown_status == 'SUCCESS':
            crown_status_str = '✓ SUCCESS'
            crown_success += 1
            total_crown_time += crown_time
        elif crown_status == 'TIMEOUT':
            crown_status_str = '⏱ TIMEOUT'
            crown_timeout += 1
        else:
            crown_status_str = f'✗ {crown_status}'
        
        if alpha_status == 'SUCCESS':
            alpha_status_str = '✓ SUCCESS'
            alpha_success += 1
            total_alpha_time += alpha_time
        elif alpha_status == 'TIMEOUT':
            alpha_status_str = '⏱ TIMEOUT'
            alpha_timeout += 1
        else:
            alpha_status_str = f'✗ {alpha_status}'
        
        crown_time_str = format_time(crown_time) if crown_time > 0 else '-'
        alpha_time_str = format_time(alpha_time) if alpha_time > 0 else '-'
        
        print(f"{r['benchmark']:<25} {crown_status_str:<20} {crown_time_str:<12} "
              f"{alpha_status_str:<20} {alpha_time_str:<12}")
    
    print("─"*90)
    print(f"\nStatistics:")
    print(f"  Total benchmarks: {len(results)}")
    print(f"  CROWN:")
    print(f"    - Successful: {crown_success}/{len(results)}")
    print(f"    - Timeouts: {crown_timeout}/{len(results)}")
    print(f"    - Total time: {format_time(total_crown_time)}")
    if crown_success > 0:
        print(f"    - Average time: {format_time(total_crown_time/crown_success)}")
    
    print(f"  alpha-CROWN:")
    print(f"    - Successful: {alpha_success}/{len(results)}")
    print(f"    - Timeouts: {alpha_timeout}/{len(results)}")
    print(f"    - Total time: {format_time(total_alpha_time)}")
    if alpha_success > 0:
        print(f"    - Average time: {format_time(total_alpha_time/alpha_success)}")
    
    # Analyze errors
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    # Categorize errors
    error_categories = {}
    
    for r in results:
        # CROWN errors
        if r.get('crown_status') != 'SUCCESS' and r.get('crown_status') != 'SKIPPED':
            error_msg = r.get('crown_error_msg', 'Unknown error')
            
            # Categorize error
            category = 'Unknown'
            if 'shape mismatch' in error_msg.lower():
                category = 'Shape mismatch'
            elif 'cannot infer' in error_msg.lower():
                category = 'Shape inference failure'
            elif 'unsupported' in error_msg.lower() or 'not supported' in error_msg.lower():
                category = 'Unsupported operation'
            elif 'lirpaerror' in error_msg.lower():
                category = 'LIRPA internal error'
            elif 'terminated' in error_msg.lower() or 'terminating' in error_msg.lower():
                category = 'Uncaught exception'
            elif 'backward' in error_msg.lower() and 'graph' in error_msg.lower():
                category = 'PyTorch autograd issue'
            elif 'inplace' in error_msg.lower():
                category = 'In-place operation issue'
            
            if category not in error_categories:
                error_categories[category] = {'crown': [], 'alpha': []}
            error_categories[category]['crown'].append(r['benchmark'])
        
        # Alpha-CROWN errors
        if r.get('alpha_status') != 'SUCCESS' and r.get('alpha_status') != 'SKIPPED' and r.get('alpha_status') != 'TIMEOUT':
            error_msg = r.get('alpha_error_msg', 'Unknown error')
            
            # Categorize error
            category = 'Unknown'
            if 'shape mismatch' in error_msg.lower():
                category = 'Shape mismatch'
            elif 'cannot infer' in error_msg.lower():
                category = 'Shape inference failure'
            elif 'unsupported' in error_msg.lower() or 'not supported' in error_msg.lower():
                category = 'Unsupported operation'
            elif 'lirpaerror' in error_msg.lower():
                category = 'LIRPA internal error'
            elif 'terminated' in error_msg.lower() or 'terminating' in error_msg.lower():
                category = 'Uncaught exception'
            elif 'backward' in error_msg.lower() and 'graph' in error_msg.lower():
                category = 'PyTorch autograd issue'
            elif 'inplace' in error_msg.lower():
                category = 'In-place operation issue'
            
            if category not in error_categories:
                error_categories[category] = {'crown': [], 'alpha': []}
            error_categories[category]['alpha'].append(r['benchmark'])
    
    if error_categories:
        print("\nError categories found:")
        for category, benchmarks in sorted(error_categories.items()):
            crown_count = len(benchmarks['crown'])
            alpha_count = len(benchmarks['alpha'])
            total = crown_count + alpha_count
            print(f"\n{category}: {total} failures ({crown_count} CROWN, {alpha_count} alpha-CROWN)")
            
            if benchmarks['crown']:
                print(f"  CROWN: {', '.join(benchmarks['crown'][:5])}")
                if len(benchmarks['crown']) > 5:
                    print(f"         ... and {len(benchmarks['crown']) - 5} more")
            
            if benchmarks['alpha']:
                print(f"  alpha-CROWN: {', '.join(benchmarks['alpha'][:5])}")
                if len(benchmarks['alpha']) > 5:
                    print(f"               ... and {len(benchmarks['alpha']) - 5} more")
    else:
        print("\nNo categorized errors found!")
    
    # Suggest fixes
    print("\n" + "─"*80)
    print("SUGGESTED FIXES:")
    print("─"*80)
    
    if 'Shape mismatch' in error_categories:
        print("\n• Shape mismatch errors:")
        print("  - These are likely bugs in CROWN backward propagation")
        print("  - Check CROWNAnalysis::addA and bias handling")
        print("  - May need to fix dimension handling for specific layer types")
    
    if 'Shape inference failure' in error_categories:
        print("\n• Shape inference failures:")
        print("  - LIRPA cannot determine proper input shapes for some layers")
        print("  - Check convolution input shape inference logic")
        print("  - May need to add shape tracking for dynamic inputs")
    
    if 'Unsupported operation' in error_categories:
        print("\n• Unsupported operations:")
        print("  - Some ONNX operations are not yet implemented")
        print("  - Check the ONNX model to see which ops are used")
        print("  - May need to add new operation handlers")
    
    if 'LIRPA internal error' in error_categories or 'Uncaught exception' in error_categories:
        print("\n• Internal errors / Uncaught exceptions:")
        print("  - These are bugs in LIRPA error handling")
        print("  - Need to add proper try-catch blocks")
        print("  - Should throw LirpaError instead of generic exceptions")
    
    if 'PyTorch autograd issue' in error_categories:
        print("\n• PyTorch autograd issues:")
        print("  - Graph reuse or in-place operation problems")
        print("  - Need to clone tensors or detach from computation graph")
        print("  - Check alpha-CROWN optimization loops")
    
    if 'In-place operation issue' in error_categories:
        print("\n• In-place operation issues:")
        print("  - PyTorch doesn't allow in-place ops during backward pass")
        print("  - Replace in-place operations with out-of-place versions")
        print("  - Use .clone() before modifying tensors")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Cleanup temporary directory
    try:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"\nWarning: Could not clean up temp directory {temp_dir}: {e}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
