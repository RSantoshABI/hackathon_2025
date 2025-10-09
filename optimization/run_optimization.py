"""Script to run price optimization using modular components."""

import pandas as pd

from data_processor import (
    load_and_filter_with_scope,
    create_elasticity_matrix,
    create_reference_arrays,
    create_bounds,
    create_segment_size_labels
)
from optimization import (
    create_evaluation_functions,
    create_objective_function,
    run_optimization,
    create_results_dataframe
)
from constraints import (
    create_constraint_functions,
    constraint_violation_detail,
    constraint_adherence,
    round_to_nearest_50
)


def run_price_optimization(
    elasticity_path, reference_path, competitor_elasticity_path,
    competitor_reference_path, seg_mapping_path, sku_detail_mapping_path,
    sku_scope_path, start_period='2025-07', end_period='2025-08',
    target_pinc=0.06, sku_lower_bound=-300, sku_upper_bound=500,
    VAT=0.19, VILC_GR=0.0378, seg_lambda=1e4, size_lambda=1e4,
    penalty_per_violation=1e3, tolerance=0.01,
    output_path='optimization_results/optimization_output.xlsx'
):
    """Run complete optimization workflow."""
    print("=" * 60)
    print("PRICE OPTIMIZATION")
    print("=" * 60)

    segment_order = ["Value", "Core", "Core+", "Premium", "Super Premium"]
    size_order = ["Small", "Regular", "Large"]

    print("\n1. Loading data...")
    sku_scope = pd.read_csv(sku_scope_path)
    own_to_own_elasticity_df = pd.read_csv(elasticity_path)
    reference_df = pd.read_csv(reference_path)
    own_to_competitor_elasticity_df = pd.read_csv(
        competitor_elasticity_path)
    competitor_reference_df = pd.read_csv(competitor_reference_path)
    seg_mapping = pd.read_csv(seg_mapping_path)
    sku_detail_mapping = pd.read_csv(sku_detail_mapping_path)

    valid_periods = reference_df['year_month'].unique()
    print(f"   Loaded data with {len(valid_periods)} periods")

    print("\n2. Preprocessing and filtering data...")
    (
        reference_df, remaining_abi_industry_volume,
        competitor_reference_df, own_to_own_elasticity_df,
        own_to_competitor_elasticity_df, reference_df_padded_bound,
        reference_df_other
    ) = load_and_filter_with_scope(
        reference_df, sku_scope, VILC_GR,
        own_to_own_elasticity_df, own_to_competitor_elasticity_df,
        competitor_reference_df, seg_mapping, start_period, end_period,
        valid_periods
    )

    print(f"   Remaining ABI volume: {remaining_abi_industry_volume:,.0f}")
    n_skus = len(reference_df['sku'].unique())
    n_months = len(reference_df['year_month'].unique())
    print(f"   Processing {n_skus} SKUs over {n_months} months")

    print("\n3. Creating elasticity matrices...")
    (
        own_products, competitor_products, months, num_months, num_own,
        num_comp, E_price_to_volume, E_price_to_comp_volume
    ) = create_elasticity_matrix(
        own_to_own_elasticity_df, reference_df,
        own_to_competitor_elasticity_df, competitor_reference_df
    )
    print(f"   Own: {num_own}, Competitor: {num_comp}, Months: {num_months}")

    print("\n4. Creating reference arrays...")
    ref_arrays = create_reference_arrays(
        reference_df, competitor_reference_df,
        months, own_products, competitor_products
    )

    M, N, K = ref_arrays['M'], ref_arrays['N'], ref_arrays['K']
    ref_price_liter = ref_arrays['ref_price_liter']
    ref_price_unit = ref_arrays['ref_price_unit']
    ref_vol_own_sellin = ref_arrays['ref_vol_own_sellin']
    ref_vol_own_sellout = ref_arrays['ref_vol_own_sellout']
    capacity = ref_arrays['capacity']
    markup = ref_arrays['markup']
    discount = ref_arrays['discount']
    excise = ref_arrays['excise']
    vilc = ref_arrays['vilc']
    ref_vol_comp = ref_arrays['ref_vol_comp']
    print(f"   Arrays created: M={M}, N={N}, K={K}")

    print("\n5. Creating evaluation functions with caching...")
    (
        evaluate_uncached, evaluate_cached, clear_eval_cache,
        base_total_MACO, NR_ref, MACO_ref
    ) = create_evaluation_functions(
        M, N, K, E_price_to_volume, E_price_to_comp_volume,
        ref_price_liter, ref_price_unit, ref_vol_own_sellin,
        ref_vol_own_sellout, ref_vol_comp, capacity,
        markup, discount, excise, vilc, VAT,
        remaining_abi_industry_volume
    )
    print(f"   Baseline MACO: {base_total_MACO:,.2f}")

    print("\n6. Creating segment and size labels...")
    segment_labels, size_labels = create_segment_size_labels(
        reference_df, M, N
    )
    print(f"   Labels created: {segment_labels.shape}")

    print("\n7. Creating objective function...")
    objective = create_objective_function(
        evaluate_cached, segment_labels, size_labels,
        segment_order, size_order, seg_lambda, size_lambda,
        penalty_per_violation
    )
    print("   Objective function created")

    print("\n8. Creating constraints...")
    constraint_dict = create_constraint_functions(
        evaluate_cached, M, N, ref_price_liter, ref_vol_own_sellin,
        ref_vol_own_sellout, ref_vol_comp, remaining_abi_industry_volume,
        target_pinc, tolerance, base_total_MACO
    )

    nl_constraints = constraint_dict['nl_constraints']
    TOTAL_REF_OWN_VOLUME = constraint_dict['TOTAL_REF_OWN_VOLUME']
    TOTAL_REF_INDUSTRY_VOLUME = (
        constraint_dict['TOTAL_REF_INDUSTRY_VOLUME']
    )
    REF_MARKET_SHARE = constraint_dict['REF_MARKET_SHARE']
    REF_AVG_PRICE_LITER = constraint_dict['REF_AVG_PRICE_LITER']
    print(f"   Created {len(nl_constraints)} constraints")

    print("\n9. Creating bounds...")
    bounds = create_bounds(
        reference_df_padded_bound, sku_lower_bound, sku_upper_bound
    )
    print(f"   Created bounds for {len(bounds)} variables")

    print("\n10. Validating bounds...")
    P0 = ref_price_unit.reshape(-1)
    p_unit_max = P0 + sku_upper_bound
    test_eval = evaluate_uncached(p_unit_max)

    max_pinc = (test_eval['avg_opt_price'] / REF_AVG_PRICE_LITER - 1)
    if max_pinc < target_pinc:
        print(
            f"   Error: Max PINC ({max_pinc:.4f}) < "
            f"target ({target_pinc:.4f})"
        )
        return None, None
    else:
        print(f"   Bounds valid (max PINC: {max_pinc:.4f})")

    print("\n11. Running optimization...")
    clear_eval_cache()

    res = run_optimization(
        objective, P0, bounds, nl_constraints,
        method='trust-constr', maxiter=5000, disp=True
    )

    print(f"\n   Optimization status: {res.success}")
    print(f"   Message: {res.message}")

    print("\n12. Checking constraint violations...")
    for idx, val, lb, ub, viol in constraint_violation_detail(
        res.x, nl_constraints
    ):
        print(
            f"   Constraint {idx}: val={val:.6g}, lb={lb:.6g}, "
            f"ub={ub:.6g}, violation={viol:.6g}"
        )

    print("\n13. Rounding prices...")
    P_opt_final = res.x.reshape(M, N)
    P_rounded = round_to_nearest_50(res.x, P0)
    P_rounded = P_rounded.reshape(M, N)

    print("\n" + "=" * 60)
    print("ORIGINAL METRICS (Before Rounding):")
    print("=" * 60)
    constraint_adherence(
        P_opt_final, evaluate_uncached, base_total_MACO,
        TOTAL_REF_OWN_VOLUME, TOTAL_REF_INDUSTRY_VOLUME,
        REF_MARKET_SHARE, target_pinc, tolerance,
        segment_labels, size_labels, segment_order, size_order, M, N
    )

    print("\n" + "=" * 60)
    print("POST ROUNDING METRICS:")
    print("=" * 60)
    constraint_adherence(
        P_rounded, evaluate_uncached, base_total_MACO,
        TOTAL_REF_OWN_VOLUME, TOTAL_REF_INDUSTRY_VOLUME,
        REF_MARKET_SHARE, target_pinc, tolerance,
        segment_labels, size_labels, segment_order, size_order, M, N
    )

    print("\n14. Creating results dataframes...")
    final_eval = evaluate_uncached(P_rounded.reshape(-1))

    results_df, industry_df = create_results_dataframe(
        P_rounded, reference_df, seg_mapping, sku_detail_mapping,
        ref_price_liter, ref_price_unit, ref_vol_own_sellin,
        NR_ref, MACO_ref, capacity, final_eval, M, N,
        ref_vol_own_sellout, competitor_reference_df, reference_df_other
    )
    print(f"   Results dataframe created with {len(results_df)} rows")
    print(f"   Industry dataframe created with {len(industry_df)} rows")

    print("\n15. Saving results...")
    results_df.to_excel(output_path, index=False)
    print(f"   Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    return res, results_df, industry_df


if __name__ == "__main__":
    base_path = (
        r'C:/Users/40107922/OneDrive - Anheuser-Busch InBev/'
        r'hackathon_2025/repo/hackathon_2025/optimization_data/'
    )

    result, results_df, industry_df = run_price_optimization(
        elasticity_path=base_path + 'elasticity.csv',
        reference_path=(
            base_path + 'reference_abi_sellin-vol_pl-ptc.csv'
        ),
        competitor_elasticity_path=base_path + 'elasticity_competitor.csv',
        competitor_reference_path=(
            base_path + 'reference_comp_sellout-vol_ptc.csv'
        ),
        seg_mapping_path=base_path + 'segment_mapping.csv',
        sku_detail_mapping_path=base_path + 'sku_details_mapping.csv',
        sku_scope_path=base_path + 'sku_scope_subset.csv',
        start_period='2025-07',
        end_period='2025-08',
        target_pinc=0.06,
        sku_lower_bound=-300,
        sku_upper_bound=500,
        output_path='optimization_results/optimization_output.xlsx'
    )

    if result:
        print("\nOptimization completed successfully!")
        print(f"  - Objective value: {result.fun:.2f}")
        print(f"  - Iterations: {result.nit}")
        print(f"  - Results shape: {results_df.shape}")
        print(f"  - Industry shape: {industry_df.shape}")
