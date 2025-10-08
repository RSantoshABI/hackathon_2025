"""
Utility functions for optimization results analysis.
"""

import numpy as np
import pandas as pd


def calculate_metrics(
    monthly_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    target_delta: float
) -> dict:
    """
    Calculate key performance metrics from optimization results.

    Args:
        monthly_df: Monthly outputs dataframe
        industry_df: Industry volumes dataframe
        target_delta: Target PINC delta

    Returns:
        Dictionary of metrics
    """
    # Volume metrics
    total_ref_volume = industry_df['volume_ref'].sum()
    total_opt_volume = industry_df['volume_opt'].sum()
    abi_ref_volume = industry_df[
        industry_df['manufacturer'] == 'abi'
    ]['volume_ref'].sum()
    abi_opt_volume = industry_df[
        industry_df['manufacturer'] == 'abi'
    ]['volume_opt'].sum()

    # MACO metrics
    total_maco_reference = monthly_df['MACO_ref'].sum()
    total_maco_optimized = monthly_df['MACO_opt'].sum()

    # Price metrics
    ref_ppl = (
        np.sum(monthly_df['volume_ref'] * monthly_df['price_liter_ref']) /
        np.sum(monthly_df['volume_ref'])
    )
    opt_ppl = (
        np.sum(monthly_df['volume_opt'] * monthly_df['price_liter_opt']) /
        np.sum(monthly_df['volume_opt'])
    )
    pinc_delta = (opt_ppl / ref_ppl) - 1

    # Market share
    ref_ms = abi_ref_volume / total_ref_volume
    opt_ms = abi_opt_volume / total_opt_volume

    return {
        'maco_reference': total_maco_reference,
        'maco_optimized': total_maco_optimized,
        'maco_change_pct': (total_maco_optimized / total_maco_reference - 1) * 100,
        'industry_volume_ref': total_ref_volume,
        'industry_volume_opt': total_opt_volume,
        'industry_volume_change_pct': (total_opt_volume / total_ref_volume - 1) * 100,
        'abi_volume_ref': abi_ref_volume,
        'abi_volume_opt': abi_opt_volume,
        'abi_volume_change_pct': (abi_opt_volume / abi_ref_volume - 1) * 100,
        'market_share_ref': ref_ms,
        'market_share_opt': opt_ms,
        'market_share_change': opt_ms - ref_ms,
        'price_per_liter_ref': ref_ppl,
        'price_per_liter_opt': opt_ppl,
        'pinc_target': target_delta,
        'pinc_achieved': pinc_delta,
        'pinc_deviation': pinc_delta - target_delta
    }


def check_hierarchy_violations(monthly_df: pd.DataFrame) -> dict:
    """
    Check for hierarchy violations in segment and size groups.

    Args:
        monthly_df: Monthly outputs dataframe

    Returns:
        Dictionary with violation information
    """
    # Segment hierarchy
    segment_order = ['Value', 'Core', 'Core+', 'Premium', 'Super Premium']
    segment_grouped = monthly_df.groupby('segment').agg({
        'NR_opt': 'sum',
        'volume_opt': 'sum'
    })
    segment_grouped['NR_per_HL'] = (
        segment_grouped['NR_opt'] / segment_grouped['volume_opt']
    )
    segment_nrh = segment_grouped['NR_per_HL'].reindex(segment_order)

    violated_segments = []
    for i in range(len(segment_order) - 1):
        if segment_nrh[segment_order[i]] > segment_nrh[segment_order[i + 1]]:
            violated_segments.append((segment_order[i], segment_order[i + 1]))

    # Size hierarchy
    size_order = ['Small', 'Regular', 'Large']
    size_grouped = monthly_df.groupby('size_group').agg({
        'NR_opt': 'sum',
        'volume_opt': 'sum'
    })
    size_grouped['NR_per_HL'] = (
        size_grouped['NR_opt'] / size_grouped['volume_opt']
    )
    size_nrh = size_grouped['NR_per_HL'].reindex(size_order)

    violated_sizes = []
    for i in range(len(size_order) - 1):
        if size_nrh[size_order[i]] < size_nrh[size_order[i + 1]]:
            violated_sizes.append((size_order[i], size_order[i + 1]))

    return {
        'segment_violations': violated_segments,
        'size_violations': violated_sizes,
        'segment_nr_per_hl': segment_nrh.to_dict(),
        'size_nr_per_hl': size_nrh.to_dict()
    }


def create_summary_report(
    monthly_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    target_delta: float,
    result
) -> pd.DataFrame:
    """
    Create a comprehensive summary report.

    Args:
        monthly_df: Monthly outputs dataframe
        industry_df: Industry volumes dataframe
        target_delta: Target PINC delta
        result: Optimization result object

    Returns:
        Summary dataframe
    """
    metrics = calculate_metrics(monthly_df, industry_df, target_delta)
    hierarchy = check_hierarchy_violations(monthly_df)

    summary_data = [
        ('Optimization', 'Success', str(result.success)),
        ('Optimization', 'Objective Value', f"{result.fun:.2f}"),
        ('Optimization', 'Iterations', str(result.nit)),
        ('', '', ''),
        ('MACO', 'Reference', f"{metrics['maco_reference']:,.0f}"),
        ('MACO', 'Optimized', f"{metrics['maco_optimized']:,.0f}"),
        ('MACO', 'Change %', f"{metrics['maco_change_pct']:.2f}%"),
        ('', '', ''),
        ('Volume', 'ABI Reference', f"{metrics['abi_volume_ref']:,.2f}"),
        ('Volume', 'ABI Optimized', f"{metrics['abi_volume_opt']:,.2f}"),
        ('Volume', 'ABI Change %', f"{metrics['abi_volume_change_pct']:.2f}%"),
        ('', '', ''),
        ('Volume', 'Industry Reference', f"{metrics['industry_volume_ref']:,.2f}"),
        ('Volume', 'Industry Optimized', f"{metrics['industry_volume_opt']:,.2f}"),
        ('Volume', 'Industry Change %', f"{metrics['industry_volume_change_pct']:.2f}%"),
        ('', '', ''),
        ('Market Share', 'Reference', f"{metrics['market_share_ref']:.4f}"),
        ('Market Share', 'Optimized', f"{metrics['market_share_opt']:.4f}"),
        ('Market Share', 'Change', f"{metrics['market_share_change']:.4f}"),
        ('', '', ''),
        ('PINC', 'Target', f"{metrics['pinc_target']:.4f}"),
        ('PINC', 'Achieved', f"{metrics['pinc_achieved']:.4f}"),
        ('PINC', 'Deviation', f"{metrics['pinc_deviation']:.4f}"),
        ('', '', ''),
        ('Hierarchy', 'Segment Violations', str(len(hierarchy['segment_violations']))),
        ('Hierarchy', 'Size Violations', str(len(hierarchy['size_violations'])))
    ]

    return pd.DataFrame(summary_data, columns=['Category', 'Metric', 'Value'])


def export_results(
    monthly_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str = 'optimization_results'
):
    """
    Export all results to Excel files.

    Args:
        monthly_df: Monthly outputs dataframe
        industry_df: Industry volumes dataframe
        summary_df: Summary dataframe
        output_dir: Output directory path
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Export with multiple sheets
    with pd.ExcelWriter(output_path / 'optimization_results.xlsx') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        monthly_df.to_excel(writer, sheet_name='Monthly Details', index=False)
        industry_df.to_excel(writer, sheet_name='Industry Volumes', index=False)

    print(f"✅ Results exported to {output_path / 'optimization_results.xlsx'}")


def compare_scenarios(
    results_dict: dict,
    scenario_names: list
) -> pd.DataFrame:
    """
    Compare multiple optimization scenarios.

    Args:
        results_dict: Dictionary of {scenario_name: (monthly_df, industry_df)}
        scenario_names: List of scenario names to compare

    Returns:
        Comparison dataframe
    """
    comparison_data = []

    for scenario_name in scenario_names:
        if scenario_name not in results_dict:
            continue

        monthly_df, industry_df = results_dict[scenario_name]
        metrics = calculate_metrics(monthly_df, industry_df, 0.06)

        comparison_data.append({
            'Scenario': scenario_name,
            'MACO': metrics['maco_optimized'],
            'MACO Change %': metrics['maco_change_pct'],
            'ABI Volume Change %': metrics['abi_volume_change_pct'],
            'Industry Volume Change %': metrics['industry_volume_change_pct'],
            'Market Share': metrics['market_share_opt'],
            'PINC Achieved': metrics['pinc_achieved']
        })

    return pd.DataFrame(comparison_data)
