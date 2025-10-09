"""
Constraints Module for Price Optimization

This module defines all optimization constraints and penalty functions.
"""

import numpy as np
from scipy.optimize import NonlinearConstraint


def penalty_ordering(
    nr, vol, segment_labels, size_labels, segment_order, size_order,
    seg_lambda=1e5, size_lambda=1e5
):
    """
    Calculate penalty for segment and size hierarchy constraint violations.
    """
    penalty = 0.0

    segment_means = {
        seg: nr[np.array(segment_labels) == seg].sum() /
        vol[np.array(segment_labels) == seg].sum()
        for seg in segment_order
        if np.any(np.array(segment_labels) == seg)
    }

    for i in range(len(segment_order) - 1):
        left, right = segment_order[i], segment_order[i+1]
        if left in segment_means and right in segment_means:
            diff = segment_means[left] - segment_means[right]
            if diff > 0:
                penalty += seg_lambda * diff

    size_means = {
        s: nr[np.array(size_labels) == s].sum() /
        vol[np.array(size_labels) == s].sum()
        for s in size_order
        if np.any(np.array(size_labels) == s)
    }
    for i in range(len(size_order) - 1):
        left, right = size_order[i], size_order[i+1]
        if left in size_means and right in size_means:
            diff = size_means[right] - size_means[left]
            if diff > 0:
                penalty += size_lambda * diff

    return penalty


def round_to_nearest_50(price_array, ref_price_array):
    """Round price differences to nearest multiple of 50."""
    price_diff = price_array - ref_price_array
    price_diff_rounded = 50 * np.round(price_diff / 50)
    return ref_price_array + price_diff_rounded


def create_constraint_functions(
    evaluate_cached, M, N, ref_price_liter, ref_vol_own_sellin,
    ref_vol_own_sellout, ref_vol_comp, remaining_abi_industry_volume,
    target_pinc, tolerance, base_total_MACO
):
    """Create all constraint functions for the optimization."""
    TOTAL_REF_INDUSTRY_VOLUME = (
        ref_vol_own_sellout.sum() +
        ref_vol_comp.sum() +
        remaining_abi_industry_volume
    )
    NEW_INDUSTRY_VOLUME = (
        (1 - 0.56 * target_pinc) * TOTAL_REF_INDUSTRY_VOLUME
    )
    TOTAL_REF_OWN_VOLUME = ref_vol_own_sellin.sum()
    REF_MARKET_SHARE = (
        (ref_vol_own_sellout.sum() + remaining_abi_industry_volume) /
        TOTAL_REF_INDUSTRY_VOLUME
    )
    REF_AVG_PRICE_LITER = (
        (ref_price_liter * ref_vol_own_sellin).sum() /
        (ref_vol_own_sellin.sum() + 1e-12)
    )

    def industry_volume_pcc_constraint(P_opt):
        """Ensure industry volume follows PCC model."""
        evals = evaluate_cached(P_opt)
        total_ind_opt = evals['total_industry_volume_opt']
        pcc_volume = (
            (1.0 - 0.56 * target_pinc) *
            TOTAL_REF_INDUSTRY_VOLUME * 0.99
        )
        return total_ind_opt - pcc_volume

    def fin_target_maco_constraint(P_opt):
        """Ensure MACO does not decrease."""
        evals = evaluate_cached(P_opt)
        return evals['total_MACO'] - base_total_MACO

    def own_volume_range(P_opt):
        """Get own volume for range constraint."""
        evals = evaluate_cached(P_opt)
        return evals['total_own_volume_opt']

    def volume_target_lower(P_opt):
        """Ensure own volume does not decrease by more than 1%."""
        evals = evaluate_cached(P_opt)
        return evals['total_own_volume_opt'] - (TOTAL_REF_OWN_VOLUME * 0.99)

    def volume_target_upper(P_opt):
        """Ensure own volume does not increase by more than 5%."""
        evals = evaluate_cached(P_opt)
        return (TOTAL_REF_OWN_VOLUME * 1.05) - evals['total_own_volume_opt']

    def pinc_constraint_target(P_opt):
        """Get average optimized price for PINC constraint."""
        evals = evaluate_cached(P_opt)
        return evals['avg_opt_price']

    def pinc_constraint_lower(P_opt):
        """Ensure PINC is not below target."""
        evals = evaluate_cached(P_opt)
        return evals['portfolio_pct_price_change'] - (target_pinc + 0.003)

    def pinc_constraint_upper(P_opt):
        """Ensure PINC is not above target."""
        evals = evaluate_cached(P_opt)
        return target_pinc + 0.003 - evals['portfolio_pct_price_change']

    def abi_share_constraint(P_opt):
        """Ensure market share does not drop by more than 0.5%."""
        evals = evaluate_cached(P_opt)
        return evals['ms_opt'] - (REF_MARKET_SHARE - 0.005)

    nl_industry_pcc = NonlinearConstraint(
        industry_volume_pcc_constraint, 0.0, np.inf
    )
    nl_maco = NonlinearConstraint(
        fin_target_maco_constraint, 0.0, np.inf
    )
    nl_vol = NonlinearConstraint(
        own_volume_range,
        0.99 * TOTAL_REF_OWN_VOLUME,
        1.05 * TOTAL_REF_OWN_VOLUME
    )
    nl_vol_low = NonlinearConstraint(volume_target_lower, 0.0, np.inf)
    nl_vol_up = NonlinearConstraint(volume_target_upper, 0.0, np.inf)
    nl_pinc = NonlinearConstraint(
        pinc_constraint_target,
        ((1 + target_pinc) * REF_AVG_PRICE_LITER) - tolerance,
        ((1 + target_pinc) * REF_AVG_PRICE_LITER) + tolerance
    )
    nl_pinc_low = NonlinearConstraint(pinc_constraint_lower, 0.0, np.inf)
    nl_pinc_up = NonlinearConstraint(pinc_constraint_upper, 0.0, np.inf)
    nl_ms = NonlinearConstraint(abi_share_constraint, 0.0, np.inf)

    constraints = [
        nl_industry_pcc, nl_maco, nl_vol_low,
        nl_vol_up, nl_pinc_low, nl_pinc_up, nl_ms
    ]
    nl_constraints = [nl_industry_pcc, nl_maco, nl_vol, nl_pinc, nl_ms]
    nl_constraints_v2 = [nl_industry_pcc, nl_maco, nl_vol, nl_ms]

    return {
        'constraints': constraints,
        'nl_constraints': nl_constraints,
        'nl_constraints_v2': nl_constraints_v2,
        'TOTAL_REF_INDUSTRY_VOLUME': TOTAL_REF_INDUSTRY_VOLUME,
        'NEW_INDUSTRY_VOLUME': NEW_INDUSTRY_VOLUME,
        'TOTAL_REF_OWN_VOLUME': TOTAL_REF_OWN_VOLUME,
        'REF_MARKET_SHARE': REF_MARKET_SHARE,
        'REF_AVG_PRICE_LITER': REF_AVG_PRICE_LITER,
        'base_total_MACO': base_total_MACO
    }


def constraint_violation_detail(res_x, constraints):
    """Calculate detailed constraint violation information."""
    details = []
    for i, c in enumerate(constraints):
        val = c.fun(res_x)
        lb = c.lb if hasattr(c, 'lb') else c.bounds[0]
        ub = c.ub if hasattr(c, 'ub') else c.bounds[1]
        v_low = max(0.0, lb - val) if lb is not None else 0.0
        v_high = max(0.0, val - ub) if ub is not None else 0.0
        violation = max(v_low, v_high)
        details.append((i, float(val), float(lb), float(ub),
                       float(violation)))
    return details


def constraint_adherence(
    P_opt_final, evaluate_uncached, base_total_MACO, TOTAL_REF_OWN_VOLUME,
    TOTAL_REF_INDUSTRY_VOLUME, REF_MARKET_SHARE, target_pinc, tolerance,
    segment_labels, size_labels, segment_order, size_order, M, N
):
    """Check and print constraint adherence for final solution."""
    final_eval = evaluate_uncached(P_opt_final.reshape(-1))

    print("MACO CONSTRAINT: ")
    print("Base total MACO:", base_total_MACO)
    print("Final total MACO:", final_eval['total_MACO'])
    print("MACO Change:", (final_eval['total_MACO'] / base_total_MACO - 1))
    if final_eval['total_MACO'] < base_total_MACO:
        print("❌ MACO declined after optimization.")
    else:
        print("✅ MACO grew after optimization.")

    print('\n')
    print("OWN VOLUME CONSTRAINT: ")
    print("Base total own volume:", TOTAL_REF_OWN_VOLUME)
    print("Final total own volume:", final_eval['total_own_volume_opt'])
    print("Own Volume Change:",
          (final_eval['total_own_volume_opt'] / TOTAL_REF_OWN_VOLUME - 1))

    lower_bound = 0.99 * TOTAL_REF_OWN_VOLUME
    upper_bound = 1.05 * TOTAL_REF_OWN_VOLUME

    if final_eval['total_own_volume_opt'] < lower_bound:
        print("❌ Constraint violated: volume decreased by more than 1%.")
    elif final_eval['total_own_volume_opt'] > upper_bound:
        print("❌ Constraint violated: volume increased by more than 5%.")
    else:
        print("✅ Constraint satisfied: volume within bounds (±1% to +5%).")

    print('\n')
    print("INDUSTRY VOLUME CONSTRAINT: ")
    print("Base industry volume:", TOTAL_REF_INDUSTRY_VOLUME)
    print("Final industry volume:", final_eval['total_industry_volume_opt'])
    vol_change = (
        final_eval['total_industry_volume_opt'] /
        TOTAL_REF_INDUSTRY_VOLUME - 1
    )
    print("Industry Volume Change:", vol_change)
    lower_bound = 0.99 * (TOTAL_REF_INDUSTRY_VOLUME * (1 - 0.56*target_pinc))

    if final_eval['total_industry_volume_opt'] < lower_bound:
        print("❌ Constraint violated: industry volume decreased >1%.")
    else:
        print("✅ Constraint satisfied: industry volume within bounds.")

    print('\n')
    print("PINC CONSTRAINT: ")
    print("Target PINC:", target_pinc)
    print("Final PINC:", final_eval['portfolio_pct_price_change'])

    if np.abs(final_eval['portfolio_pct_price_change'] - target_pinc
              ) > tolerance:
        print(f"❌ Constraint violated: PINC > {target_pinc}")
    else:
        print(f"✅ Constraint satisfied: PINC = {target_pinc}")

    print('\n')
    print("MARKET SHARE CONSTRAINT: ")
    print("Base Market Share:", REF_MARKET_SHARE)
    print("Final Market Share:", final_eval['ms_opt'])

    bound = REF_MARKET_SHARE - 0.005

    if final_eval['ms_opt'] < bound:
        print("❌ Constraint violated: market share decreased by >0.5%.")
    else:
        print("✅ Constraint satisfied: market share within bounds.")

    segment_means = {
        seg: (
            final_eval['NR_opt'][np.array(segment_labels) == seg].sum() /
            final_eval['Q_own_opt'][np.array(segment_labels) == seg].sum()
        )
        for seg in segment_order
        if np.any(np.array(segment_labels) == seg)
    }

    size_means = {
        s: (
            final_eval['NR_opt'][np.array(size_labels) == s].sum() /
            final_eval['Q_own_opt'][np.array(size_labels) == s].sum()
        )
        for s in size_order
        if np.any(np.array(size_labels) == s)
    }

    print("\n")
    print("HIERARCHY CONSTRAINTS: ")
    violated_segments = []
    for i in range(len(segment_order) - 1):
        curr_seg = segment_order[i]
        next_seg = segment_order[i + 1]
        if (next_seg in segment_means.keys() and
                curr_seg in segment_means.keys()):
            if segment_means[curr_seg] > segment_means[next_seg]:
                violated_segments.append((curr_seg, next_seg))

    if violated_segments:
        print("❌ Segment NR/HL hierarchy violated between:")
        for seg1, seg2 in violated_segments:
            print(f"  {seg1} > {seg2}")
    else:
        print("✅ Segment NR/HL hierarchy is satisfied")

    violated_sizes = []
    for i in range(len(size_order) - 1):
        curr_size = size_order[i]
        next_size = size_order[i + 1]
        if (next_size in size_means.keys() and
                curr_size in size_means.keys()):
            if size_means[curr_size] < size_means[next_size]:
                violated_sizes.append((curr_size, next_size))

    if violated_sizes:
        print("❌ Size group NR/HL hierarchy violated between:")
        for sg1, sg2 in violated_sizes:
            print(f"  {sg1} < {sg2}")
    else:
        print("✅ Size group NR/HL hierarchy is satisfied")
