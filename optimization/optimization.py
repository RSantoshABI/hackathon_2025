"""
Optimization Module for Price Optimization

This module contains the core optimization logic including evaluation
functions with caching mechanism for performance optimization.
"""

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize


_eval_cache = {}


def create_evaluation_functions(
    M, N, K, E_price_to_volume, E_price_to_comp_volume,
    ref_price_liter, ref_price_unit, ref_vol_own_sellin,
    ref_vol_own_sellout, ref_vol_comp, capacity,
    markup, discount, excise, vilc, VAT,
    remaining_abi_industry_volume
):
    """
    Create evaluation functions with caching mechanism.

    Args:
        M: Number of months
        N: Number of own products
        K: Number of competitor products
        E_price_to_volume: Own-to-own elasticity matrix (N x N)
        E_price_to_comp_volume: Own-to-competitor elasticity matrix (N x K)
        ref_price_liter: Reference prices per liter (M x N)
        ref_price_unit: Reference prices per unit (M x N)
        ref_vol_own_sellin: Reference own volumes sell-in (M x N)
        ref_vol_own_sellout: Reference own volumes sell-out (M x N)
        ref_vol_comp: Reference competitor volumes (M x K)
        capacity: Product capacities (M x N)
        markup: Markup values (M x N)
        discount: Discount values (M x N)
        excise: Excise values (M x N)
        vilc: VILC values (M x N)
        VAT: VAT rate
        remaining_abi_industry_volume: Volume from other ABI products

    Returns:
        Tuple of evaluation functions and baseline metrics
    """
    total_ref_industry_volume = (
        ref_vol_own_sellout.sum() +
        ref_vol_comp.sum() +
        remaining_abi_industry_volume
    )
    total_ref_own_volume = ref_vol_own_sellin.sum()
    total_ref_own_volume_sellout = (
        ref_vol_own_sellout.sum() + remaining_abi_industry_volume
    )

    NR_per_unit_ref = (
        ((ref_price_unit) / (1 + markup)) *
        (1 + discount + excise)
    ) / (1 + VAT)
    sales_unit_ref = (ref_vol_own_sellin * 100000) / capacity
    NR_ref = NR_per_unit_ref * sales_unit_ref
    MACO_ref = NR_ref + vilc
    base_total_MACO = MACO_ref.sum()

    def evaluate_uncached(P_opt):
        """Evaluate objective and constraints for given prices (uncached)."""
        t0 = time.time()
        P = P_opt.reshape(M, N)
        P_L = P * 1000 / capacity

        safe_ref = np.where(ref_price_liter <= 0, 1e-8, ref_price_liter)
        log_price_ratio = np.log(np.maximum(P_L, 1e-8) / safe_ref)

        exponent_own = log_price_ratio.dot(E_price_to_volume)
        Q_own_opt = ref_vol_own_sellin * np.exp(exponent_own)
        Q_own_opt_sellout = ref_vol_own_sellout * np.exp(exponent_own)

        exponent_comp = log_price_ratio.dot(E_price_to_comp_volume)
        Q_comp_opt = ref_vol_comp * np.exp(exponent_comp)

        total_industry_volume_opt = (
            Q_own_opt_sellout.sum() +
            Q_comp_opt.sum() +
            remaining_abi_industry_volume
        )
        total_own_volume_opt_sellout = (
            Q_own_opt_sellout.sum() + remaining_abi_industry_volume
        )
        total_own_volume_opt = Q_own_opt.sum()

        avg_ref_price = (
            (ref_price_liter * ref_vol_own_sellin).sum() /
            (ref_vol_own_sellin.sum() + 1e-12)
        )
        avg_opt_price = (
            (P_L * Q_own_opt).sum() / (Q_own_opt.sum() + 1e-12)
        )
        portfolio_pct_price_change = avg_opt_price / avg_ref_price - 1.0

        ms_ref = total_ref_own_volume_sellout / total_ref_industry_volume
        ms_opt = total_own_volume_opt_sellout / total_industry_volume_opt

        NR_per_unit_opt = (
            ((P) / (1 + markup)) * (1 + discount + excise)
        ) / (1 + VAT)
        sales_unit_opt = (Q_own_opt * 100000) / capacity
        NR_opt = NR_per_unit_opt * sales_unit_opt
        MACO_opt = NR_opt + vilc

        total_MACO = MACO_opt.sum()

        dt = time.time() - t0
        return dict(
            P=P,
            log_price_ratio=log_price_ratio,
            Q_own_opt=Q_own_opt,
            Q_own_opt_sellout=Q_own_opt_sellout,
            Q_comp_opt=Q_comp_opt,
            total_industry_volume_opt=total_industry_volume_opt,
            total_own_volume_opt=total_own_volume_opt,
            total_ref_industry_volume=total_ref_industry_volume,
            total_ref_own_volume=total_ref_own_volume,
            ms_ref=ms_ref,
            ms_opt=ms_opt,
            total_MACO=total_MACO,
            portfolio_pct_price_change=portfolio_pct_price_change,
            avg_opt_price=avg_opt_price,
            NR_opt=NR_opt,
            MACO_opt=MACO_opt,
            eval_time=dt,
        )

    def evaluate_cached(P_opt):
        """Evaluate with caching."""
        key = P_opt.tobytes()
        out = _eval_cache.get(key)
        if out is None:
            out = evaluate_uncached(P_opt)
            _eval_cache[key] = out
        return out

    def clear_eval_cache():
        """Clear the evaluation cache."""
        global _eval_cache
        _eval_cache = {}

    return (evaluate_uncached, evaluate_cached, clear_eval_cache,
            base_total_MACO, NR_ref, MACO_ref)


def create_objective_function(
    evaluate_cached, segment_labels, size_labels,
    segment_order, size_order, seg_lambda, size_lambda,
    penalty_per_violation
):
    """Create the objective function for optimization."""
    from constraints import penalty_ordering

    def objective(P_opt):
        """Objective function: maximize MACO with hierarchy penalties."""
        evals = evaluate_cached(P_opt)
        total_MACO = evals['total_MACO']
        NR_opt_array = evals['NR_opt']
        vol_opt_array = evals['Q_own_opt']

        base_obj = - total_MACO

        hierarchy_penalty = penalty_ordering(
            NR_opt_array,
            vol_opt_array,
            segment_labels,
            size_labels,
            segment_order,
            size_order,
            seg_lambda=seg_lambda,
            size_lambda=size_lambda
        )

        return base_obj + hierarchy_penalty

    return objective


def run_optimization(
    objective, P0, bounds, constraints, method='trust-constr',
    maxiter=5000, disp=True
):
    """Run the price optimization."""
    t_start = time.time()

    res = minimize(
        objective,
        P0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={
            'disp': disp,
            "maxiter": maxiter
        }
    )

    t_taken = time.time() - t_start
    msg = "Optimization finished in {:.1f}s, success={}, message={}"
    print(msg.format(t_taken, res.success, res.message))

    return res


def create_results_dataframe(
    P_rounded, reference_df, seg_mapping, sku_detail_mapping,
    ref_price_liter, ref_price_unit, ref_vol_own_sellin,
    NR_ref, MACO_ref, capacity, final_eval, M, N,
    ref_vol_own_sellout, competitor_reference_df, reference_df_other
):
    """Create results dataframe and industry dataframe with reference
    and optimized values."""
    sku_list = np.array(reference_df["sku"].unique())
    month_list = np.sort(reference_df["year_month"].unique())

    n_skus = len(sku_list)
    n_months = len(month_list)

    sku_repeat = np.tile(sku_list, n_months)
    month_tile = np.repeat(month_list, n_skus)

    results_df = pd.DataFrame({
        "sku": sku_repeat,
        "year_month": month_tile,
        "price_liter_ref": ref_price_liter.flatten(),
        "price_unit_ref": ref_price_unit.flatten(),
        "volume_ref": ref_vol_own_sellin.flatten(),
        "NR_ref": NR_ref.flatten(),
        "MACO_ref": MACO_ref.flatten(),
        "price_liter_opt": P_rounded.flatten() * 1000 / capacity.flatten(),
        "price_unit_opt": P_rounded.flatten(),
        "volume_opt": final_eval['Q_own_opt'].flatten(),
        "NR_opt": final_eval['NR_opt'].flatten(),
        "MACO_opt": final_eval['MACO_opt'].flatten()
    })

    industry_abi_df = pd.DataFrame({
        "sku": sku_repeat,
        "year_month": month_tile,
        "manufacturer": "ABI",
        "volume_ref": ref_vol_own_sellout.flatten(),
        "volume_opt": final_eval['Q_own_opt'].flatten(),
    })

    # Create dataframe for competitor volumes
    comp_sku_list = np.array(competitor_reference_df["sku"].unique())
    comp_month_list = np.sort(competitor_reference_df["year_month"].unique())

    comp_n_skus = len(comp_sku_list)
    comp_n_months = len(comp_month_list)

    comp_sku_repeat = np.tile(comp_sku_list, comp_n_months)
    comp_month_tile = np.repeat(comp_month_list, comp_n_skus)

    # Get reference volumes in correct order
    comp_ref_vols = []
    for sku in comp_sku_list:
        for month in comp_month_list:
            vol = competitor_reference_df[
                (competitor_reference_df['sku'] == sku) &
                (competitor_reference_df['year_month'] == month)
            ]['reference_volume'].values
            comp_ref_vols.append(vol[0] if len(vol) > 0 else 0)

    industry_comp_df = pd.DataFrame({
        "sku": comp_sku_repeat,
        "year_month": comp_month_tile,
        "manufacturer": "Competitor",
        "volume_ref": comp_ref_vols,
        "volume_opt": final_eval['Q_comp_opt'].flatten(),
    })

    # Create dataframe for other ABI SKUs (not in optimization scope)
    if len(reference_df_other) > 0:
        abi_other_comp_df = reference_df_other[[
            'sku', 'year_month', 'sellout_volume'
        ]].copy()
        abi_other_comp_df['manufacturer'] = 'ABI'
        abi_other_comp_df['volume_ref'] = abi_other_comp_df['sellout_volume']
        abi_other_comp_df['volume_opt'] = abi_other_comp_df['sellout_volume']
        abi_other_comp_df = abi_other_comp_df[[
            'sku', 'year_month', 'manufacturer', 'volume_ref', 'volume_opt'
        ]]
    else:
        abi_other_comp_df = pd.DataFrame(columns=[
            'sku', 'year_month', 'manufacturer', 'volume_ref', 'volume_opt'
        ])

    industry_df = pd.concat([
        industry_abi_df, industry_comp_df, abi_other_comp_df
    ])

    results_df = pd.merge(results_df, seg_mapping, on=['sku'], how='left')
    results_df = pd.merge(
        results_df,
        sku_detail_mapping[sku_detail_mapping.columns[:-1]],
        on=['sku'],
        how='left'
    )
    
    # Create size_group based on capacity
    if 'capacity' in results_df.columns:
        def get_size_group(cap):
            if cap < 0.5:
                return 'Small'
            elif cap < 1.0:
                return 'Regular'
            else:
                return 'Large'
        
        results_df['size_group'] = results_df['capacity'].apply(get_size_group)

    return results_df, industry_df
