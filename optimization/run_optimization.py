"""
Example script showing how to use the optimization modules programmatically.
This can be used as a standalone script or imported into other projects.
"""

import numpy as np
from pathlib import Path

from data_processor import DataProcessor
from constraints import ConstraintManager
from optimization import PriceOptimizer
from utils import create_summary_report, export_results


def run_optimization(
    elasticity_path: str,
    reference_path: str,
    competitor_elasticity_path: str,
    competitor_reference_path: str,
    seg_mapping_path: str,
    start_period: str = '2025-08',
    end_period: str = '2025-10',
    target_delta: float = 0.06,
    VAT: float = 0.19,
    VILC_GR: float = 0.0378,
    tolerance: float = 0.005,
    output_dir: str = 'optimization_results'
):
    """
    Run complete optimization workflow.

    Args:
        elasticity_path: Path to own-to-own elasticity CSV
        reference_path: Path to reference data CSV
        competitor_elasticity_path: Path to competitor elasticity CSV
        competitor_reference_path: Path to competitor reference CSV
        seg_mapping_path: Path to segment mapping CSV
        start_period: Optimization start period (YYYY-MM)
        end_period: Optimization end period (YYYY-MM)
        target_delta: Target portfolio PINC
        VAT: VAT rate
        VILC_GR: VILC growth rate
        n_jobs: Number of parallel jobs (-1 for all cores)
        tolerance: PINC constraint tolerance
        output_dir: Directory for output files

    Returns:
        Tuple of (result, monthly_df, industry_df, summary_df)
    """
    print("="*60)
    print("PRICE OPTIMIZATION")
    print("="*60)

    # Step 1: Data Processing
    print("\n1. Loading and processing data...")
    data_processor = DataProcessor(
        min_period='2024-01',
        max_period='2026-12',
        vilc_gr=VILC_GR
    )

    (reference_df, competitor_reference_df, E_price_to_volume,
     E_price_to_comp_volume, metadata) = data_processor.load_and_process_data(
        elasticity_path=elasticity_path,
        reference_path=reference_path,
        competitor_elasticity_path=competitor_elasticity_path,
        competitor_reference_path=competitor_reference_path,
        seg_mapping_path=seg_mapping_path,
        start_period=start_period,
        end_period=end_period
    )

    print(f"   ✓ Loaded {metadata['num_own']} own products")
    print(f"   ✓ Loaded {metadata['num_comp']} competitor products")
    print(f"   ✓ Processing {metadata['num_months']} months")

    # Step 2: Initialize Optimizer
    print("\n2. Initializing optimizer...")
    optimizer = PriceOptimizer(
        reference_df=reference_df,
        competitor_reference_df=competitor_reference_df,
        own_products=metadata['own_products'],
        competitor_products=metadata['competitor_products'],
        months=metadata['months'],
        num_months=metadata['num_months'],
        num_own=metadata['num_own'],
        num_comp=metadata['num_comp'],
        E_price_to_volume=E_price_to_volume,
        E_price_to_comp_volume=E_price_to_comp_volume,
        VAT=VAT,
    )
    print(f"   ✓ Optimizer ready!")

    # Step 3: Create Constraints
    print("\n3. Setting up constraints...")
    constraint_manager = ConstraintManager(
        reference_df=reference_df,
        competitor_reference_df=competitor_reference_df,
        own_products=metadata['own_products'],
        competitor_products=metadata['competitor_products'],
        months=metadata['months'],
        num_months=metadata['num_months'],
        num_own=metadata['num_own'],
        num_comp=metadata['num_comp'],
        E_price_to_volume=E_price_to_volume,
        E_price_to_comp_volume=E_price_to_comp_volume,
        target_delta=target_delta,
        VAT=VAT,
        get_reference_arrays_own_func=optimizer.get_reference_arrays_own,
        get_reference_arrays_comp_func=optimizer.get_reference_arrays_comp,
        calc_volume_func=optimizer.calc_volume,
        calc_MACO_func=optimizer.calc_MACO
    )

    constraints = constraint_manager.create_constraints(tolerance=tolerance)
    print(f"   ✓ Created {len(constraints)} constraints")

    # Step 4: Setup Bounds and Initial Guess
    print("\n4. Preparing optimization parameters...")
    bounds = optimizer.create_bounds(metadata['reference_df_padded_bound'])

    reference_df_ordered = reference_df.set_index(['sku', 'year_month']).loc[
        [(sku, month) for month in metadata['months']
         for sku in metadata['own_products']]
    ].reset_index()

    reference_df_ordered['reference_price_unit'] = (
        reference_df_ordered['reference_price'] *
        reference_df_ordered['capacity'] / 1000
    )

    P0 = reference_df_ordered['reference_price_unit'].values
    print(f"   ✓ Created bounds for {len(bounds)} variables")
    print(f"   ✓ Initial guess shape: {P0.shape}")

    # Step 5: Run Optimization
    print("\n5. Running optimization...")
    print("   (This may take several minutes...)")

    (result, monthly_outputs_unrounded, industry_volume_unrounded,
     rounded_prices, monthly_outputs_rounded,
     industry_volume_rounded) = optimizer.optimize(
        constraints=constraints,
        bounds=bounds,
        P0=P0,
        method='trust-constr',
        options={'disp': True}
    )

    print(f"\n   ✓ Optimization complete!")
    print(f"   ✓ Success: {result.success}")
    print(f"   ✓ Objective value: {result.fun:.2f}")
    print(f"   ✓ Iterations: {result.nit}")

    # Step 6: Create Summary
    print("\n6. Creating summary report...")
    summary_df = create_summary_report(
        monthly_outputs_rounded,
        industry_volume_rounded,
        target_delta,
        result
    )
    print("   ✓ Summary report created")

    # Step 7: Export Results
    print("\n7. Exporting results...")
    export_results(
        monthly_outputs_rounded,
        industry_volume_rounded,
        summary_df,
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

    return result, monthly_outputs_rounded, industry_volume_rounded, summary_df


if __name__ == "__main__":
    # Example usage
    DATA_DIR = r'C:/Users/40107922/Downloads/r2'

    result, monthly_df, industry_df, summary_df = run_optimization(
        elasticity_path=f'{DATA_DIR}/elasticity.csv',
        reference_path=f'{DATA_DIR}/subset_reference_abi_sellin-vol_pl-ptc.csv',
        competitor_elasticity_path=f'{DATA_DIR}/elasticity_competitor.csv',
        competitor_reference_path=f'{DATA_DIR}/subset_reference_comp_sellout-vol_ptc.csv',
        seg_mapping_path=f'{DATA_DIR}/segment_mapping.csv',
        start_period='2025-08',
        end_period='2025-10',
        target_delta=0.06,
        n_jobs=-1,
        output_dir='optimization_results'
    )

    print("\nResults available in variables:")
    print("  - result: Optimization result object")
    print("  - monthly_df: Monthly optimization details")
    print("  - industry_df: Industry volume details")
    print("  - summary_df: Summary report")
