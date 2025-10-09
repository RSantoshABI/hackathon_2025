"""
Data Processing Module for Price Optimization

This module handles all data preprocessing, validation, and transformation
required for the price optimization model. It maintains the exact functionality
from the original notebook implementation.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def load_and_filter_with_scope(
    reference_df, sku_scope_df, VILC_GR,
    own_to_own_elasticity_df, own_to_competitor_elasticity_df,
    competitor_reference_df, seg_mapping, start_period, end_period,
    valid_periods
):
    """
    Load data and filter based on SKU scope, calculating remaining volume.

    Args:
        reference_df: Full reference dataframe
        sku_scope_df: DataFrame with SKU scope subset
        VILC_GR: VILC growth rate
        own_to_own_elasticity_df: Own-to-own elasticity data
        own_to_competitor_elasticity_df: Own-to-competitor elasticity data
        competitor_reference_df: Competitor reference data
        seg_mapping: Segment mapping data
        start_period: Start period string
        end_period: End period string
        valid_periods: Valid periods array

    Returns:
        tuple: (filtered reference_df, remaining_abi_industry_volume,
                preprocessed dataframes, reference_df_other)
    """
    reference_df_full = reference_df.copy()
    reference_df_full, _, _, _, _ = pre_processor(
        own_to_own_elasticity_df, reference_df,
        own_to_competitor_elasticity_df, competitor_reference_df,
        seg_mapping, start_period, end_period, valid_periods, VILC_GR
    )

    reference_df_other = reference_df_full[
        ~reference_df_full['sku'].isin(sku_scope_df['sku'].unique())
    ]
    remaining_abi_industry_volume = (
        reference_df_other['sellout_volume'].sum()
    )

    reference_df_filtered = reference_df[
        reference_df['sku'].isin(sku_scope_df['sku'].unique())
    ]

    (
        reference_df_processed, competitor_reference_df_processed,
        own_to_own_elasticity_df_processed,
        own_to_competitor_elasticity_df_processed,
        reference_df_padded_bound
    ) = pre_processor(
        own_to_own_elasticity_df, reference_df_filtered,
        own_to_competitor_elasticity_df, competitor_reference_df,
        seg_mapping, start_period, end_period, valid_periods, VILC_GR
    )

    return (
        reference_df_processed, remaining_abi_industry_volume,
        competitor_reference_df_processed,
        own_to_own_elasticity_df_processed,
        own_to_competitor_elasticity_df_processed,
        reference_df_padded_bound,
        reference_df_other
    )


def generate_periods(start_period, end_period):
    """
    Generate list of periods between start and end dates.

    Args:
        start_period: Start date in format 'YYYY-MM'
        end_period: End date in format 'YYYY-MM'

    Returns:
        List of periods in 'YYYY-MM' format
    """
    start_date = datetime.strptime(start_period, "%Y-%m")
    end_date = datetime.strptime(end_period, "%Y-%m")
    periods = []
    current_date = start_date
    while current_date <= end_date:
        periods.append(current_date.strftime("%Y-%m"))
        next_month = current_date.replace(day=28) + pd.DateOffset(days=4)
        current_date = next_month.replace(day=1)
    return periods


def adjust_period(period, years_to_subtract):
    """
    Adjust period by subtracting specified number of years.

    Args:
        period: Period in 'YYYY-MM' format
        years_to_subtract: Number of years to subtract

    Returns:
        Adjusted period in 'YYYY-MM' format
    """
    period_date = datetime.strptime(period, "%Y-%m")
    adjusted_date = period_date.replace(year=period_date.year - years_to_subtract)
    return adjusted_date.strftime("%Y-%m")


def check_periods_in_valid_range(periods, valid_periods):
    """
    Check if periods exist in valid range, adjust if necessary using historical data.

    Args:
        periods: List of periods to check
        valid_periods: List of valid periods available in data

    Returns:
        Tuple of (adjusted_periods, period_mapping)
    """
    adjusted_periods = []
    period_mapping = {}
    for period in periods:
        original_period = period
        adjusted = period
        if period in valid_periods:
            adjusted_periods.append(adjusted)
            period_mapping[original_period] = adjusted
        else:
            adjusted_period_1 = adjust_period(period, 1)
            if adjusted_period_1 in valid_periods:
                adjusted_periods.append(adjusted_period_1)
                period_mapping[original_period] = adjusted_period_1
            else:
                adjusted_period_2 = adjust_period(adjusted_period_1, 1)
                if adjusted_period_2 in valid_periods:
                    adjusted_periods.append(adjusted_period_2)
                    period_mapping[original_period] = adjusted_period_2
                else:
                    raise ValueError(f"Period {period} is invalid")
    return adjusted_periods, period_mapping


def extract_pack_type(sku):
    """
    Extract pack type from SKU string.

    Args:
        sku: SKU identifier string

    Returns:
        Pack type ('NRB', 'RB', 'CAN', or 'UNKNOWN')
    """
    known_pack_types = ['NRB', 'RB', 'CAN']
    for pt in known_pack_types:
        if pt in sku:
            return pt
    return 'UNKNOWN'


def assign_size_group(row):
    """
    Assign size group based on capacity and pack type.

    Args:
        row: DataFrame row with 'capacity' and 'pack_type' columns

    Returns:
        Size group ('Small', 'Regular', 'Large', or 'Unknown')
    """
    cap = row['capacity']
    pt = row['pack_type']

    if cap < 300:
        return 'Small'
    elif pt == 'CAN':
        if 300 <= cap <= 399:
            return 'Regular'
        elif cap > 399:
            return 'Large'
    elif pt in ['RB', 'NRB']:
        if 300 <= cap <= 599:
            return 'Regular'
        elif cap > 599:
            return 'Large'
    return 'Unknown'


def pre_processor(own_to_own_elasticity_df, reference_df, own_to_competitor_elasticity_df, 
                  competitor_reference_df, seg_mapping, start_period, end_period, 
                  valid_periods, VILC_GR):
    """
    Preprocess all data for optimization including period adjustment, padding, and enrichment.

    Args:
        own_to_own_elasticity_df: Own product cross-elasticity matrix
        reference_df: Reference data for own products
        own_to_competitor_elasticity_df: Own to competitor elasticity matrix
        competitor_reference_df: Reference data for competitor products
        seg_mapping: Segment mapping data
        start_period: Optimization start period
        end_period: Optimization end period
        valid_periods: Available periods in historical data
        VILC_GR: VILC annual growth rate

    Returns:
        Tuple of processed dataframes
    """
    periods_to_check = generate_periods(start_period, end_period)
    adjusted_periods, period_mapping = check_periods_in_valid_range(periods_to_check, valid_periods)

    # Process reference dataframe
    reference_df_filt = reference_df[reference_df['year_month'].isin(adjusted_periods)].copy()
    period_map_rev = {v: k for k, v in period_mapping.items()}
    reference_df_filt['year_month_map'] = reference_df_filt['year_month'].map(period_map_rev)
    reference_df_filt['used_prior'] = (reference_df_filt['year_month'] != reference_df_filt['year_month_map'])
    reference_df_filt.loc[reference_df_filt['used_prior'], 'vilc'] = reference_df_filt.loc[reference_df_filt['used_prior'], 'vilc'] * (1 + VILC_GR)
    reference_df_filt.rename(columns={'year_month': 'year_month_og'}, inplace=True)
    reference_df_filt.rename(columns={'year_month_map': 'year_month'}, inplace=True)
    reference_df_filt.drop(columns={'year_month_og'}, inplace=True)
    cols = reference_df_filt.columns[:-2]
    cols = cols.insert(0, reference_df_filt.columns[-2])
    reference_df_filt = reference_df_filt[cols]
    reference_df_filt = reference_df_filt.sort_values(by=['year_month', 'sku'])

    # Process competitor reference dataframe
    competitor_reference_df_filt = competitor_reference_df[competitor_reference_df['year_month'].isin(adjusted_periods)].copy()
    competitor_reference_df_filt['year_month_map'] = competitor_reference_df_filt['year_month'].map(period_map_rev)
    competitor_reference_df_filt['used_prior'] = (competitor_reference_df_filt['year_month'] != competitor_reference_df_filt['year_month_map'])
    competitor_reference_df_filt.rename(columns={'year_month': 'year_month_og'}, inplace=True)
    competitor_reference_df_filt.rename(columns={'year_month_map': 'year_month'}, inplace=True)
    competitor_reference_df_filt.drop(columns={'year_month_og'}, inplace=True)
    cols = competitor_reference_df_filt.columns[:-2]
    cols = cols.insert(0, competitor_reference_df_filt.columns[-2])
    competitor_reference_df_filt = competitor_reference_df_filt[cols]
    competitor_reference_df_filt = competitor_reference_df_filt.sort_values(by=['year_month', 'sku'])

    reference_df = reference_df_filt.copy()
    competitor_reference_df = competitor_reference_df_filt.copy()

    reference_df = reference_df.reset_index(drop=True)
    competitor_reference_df = competitor_reference_df.reset_index(drop=True)

    # Filter products by elasticity coverage
    own_products_init = reference_df['sku'].unique()
    competitor_products_init = competitor_reference_df['sku'].unique()

    own_to_own_elasticity_df = own_to_own_elasticity_df[own_to_own_elasticity_df['target_sku'].isin(own_products_init)]
    own_to_own_elasticity_df = own_to_own_elasticity_df[own_to_own_elasticity_df['other_sku'].isin(own_products_init)]

    own_to_competitor_elasticity_df = own_to_competitor_elasticity_df[own_to_competitor_elasticity_df['target_sku'].isin(own_products_init)]
    own_to_competitor_elasticity_df = own_to_competitor_elasticity_df[own_to_competitor_elasticity_df['other_sku'].isin(competitor_products_init)]

    reference_df = reference_df[['year_month', 'sku', 'reference_volume', 'reference_price', 'markup', 'discount', 'excise', 'vilc', 'sellout_volume']]

    # Extract unique SKUs and periods
    own_skus = own_to_own_elasticity_df['target_sku'].unique()
    own_skus.sort()
    competitor_skus = own_to_competitor_elasticity_df['other_sku'].unique()
    competitor_skus.sort()

    all_months = reference_df['year_month'].unique()
    all_months_date = pd.to_datetime(all_months)
    all_months_date = pd.Series(all_months_date).sort_values().unique()

    # Pad own products reference data
    full_index_own = pd.MultiIndex.from_product(
        [all_months_date, own_skus],
        names=['year_month_date', 'sku']
    )

    reference_df['year_month_date'] = pd.to_datetime(reference_df['year_month'])
    reference_df = reference_df.set_index(['year_month_date', 'sku'])
    reference_df_own_padded = reference_df.reindex(full_index_own)

    reference_df_own_padded['present'] = 'present'
    reference_df_own_padded.loc[reference_df_own_padded['reference_volume'].isna(), 'present'] = 'missing'

    reference_df_own_padded['reference_volume'] = reference_df_own_padded['reference_volume'].fillna(0)
    reference_df_own_padded['sellout_volume'] = reference_df_own_padded['sellout_volume'].fillna(0)

    mean_prices = reference_df['reference_price'].mean()
    reference_df_own_padded['reference_price'] = reference_df_own_padded['reference_price'].fillna(mean_prices)

    for col in ['markup', 'discount', 'excise', 'vilc']:
        if col in reference_df_own_padded.columns:
            reference_df_own_padded[col] = reference_df_own_padded[col].fillna(0)
    
    reference_df_own_padded = reference_df_own_padded.reset_index()
    
    year_month_mapping = reference_df.reset_index()
    year_month_mapping = year_month_mapping[['year_month_date', 'year_month']].drop_duplicates()
    
    reference_df_own_padded = pd.merge(reference_df_own_padded, year_month_mapping, how='left', on='year_month_date')
    
    reference_df_own_padded = reference_df_own_padded.drop(columns=['year_month_date', 'year_month_x'])
    reference_df_own_padded.rename(columns={'year_month_y': 'year_month'}, inplace=True)
    reference_df_own_padded = pd.merge(reference_df_own_padded, seg_mapping, how='left', on='sku')
    reference_df_padded_bound = reference_df_own_padded.copy()
    reference_df_own_padded = reference_df_own_padded.drop(columns={'present'})

    # Pad competitor reference data
    all_months = competitor_reference_df['year_month'].unique()
    all_months_date = pd.to_datetime(all_months)
    all_months_date = pd.Series(all_months_date).sort_values().unique()
    
    full_index_comp = pd.MultiIndex.from_product(
        [all_months_date, competitor_skus],
        names=['year_month_date', 'sku']
    )
    
    competitor_reference_df['year_month_date'] = pd.to_datetime(competitor_reference_df['year_month'])
    competitor_reference_df = competitor_reference_df.set_index(['year_month_date', 'sku'])
    comp_reference_df_own_padded = competitor_reference_df.reindex(full_index_comp)
    
    comp_reference_df_own_padded['reference_volume'] = comp_reference_df_own_padded['reference_volume'].fillna(0)
    
    mean_prices = comp_reference_df_own_padded['reference_price'].mean()
    comp_reference_df_own_padded['reference_price'] = comp_reference_df_own_padded['reference_price'].fillna(mean_prices)
    
    comp_reference_df_own_padded = comp_reference_df_own_padded.reset_index()
    
    year_month_mapping = competitor_reference_df.reset_index()
    year_month_mapping = year_month_mapping[['year_month_date', 'year_month']].drop_duplicates()
    
    comp_reference_df_own_padded = pd.merge(comp_reference_df_own_padded, year_month_mapping, how='left', on='year_month_date')
    comp_reference_df_own_padded = comp_reference_df_own_padded.drop(columns=['year_month_date', 'year_month_x'])
    comp_reference_df_own_padded.rename(columns={'year_month_y': 'year_month'}, inplace=True)
    
    reference_df = reference_df_own_padded.copy()
    competitor_reference_df = comp_reference_df_own_padded.copy()

    # Extract pack type and assign size groups
    reference_df['pack_type'] = reference_df['sellin_sku'].apply(extract_pack_type)
    reference_df.loc[(reference_df['pack_type'] == 'UNKNOWN') & (reference_df['sku'].str.contains('NO RETORNABLE')), 'pack_type'] = 'NRB'
    reference_df.loc[(reference_df['pack_type'] == 'UNKNOWN') & (reference_df['sku'].str.contains('RETORNABLE')), 'pack_type'] = 'RB'

    reference_df['size_group'] = reference_df.apply(assign_size_group, axis=1)
    reference_df['reference_price_unit'] = reference_df['reference_price'] * reference_df['capacity'] / 1000

    return reference_df, competitor_reference_df, own_to_own_elasticity_df, own_to_competitor_elasticity_df, reference_df_padded_bound


def create_elasticity_matrix(own_to_own_elasticity_df, reference_df, own_to_competitor_elasticity_df, competitor_reference_df):
    """
    Create elasticity matrices for optimization.
    
    Args:
        own_to_own_elasticity_df: Own product cross-elasticity data
        reference_df: Reference data for own products
        own_to_competitor_elasticity_df: Own to competitor elasticity data
        competitor_reference_df: Reference data for competitor products
    
    Returns:
        Tuple containing product lists, time periods, dimensions, and elasticity matrices
    """
    own_products = reference_df['sku'].unique()
    competitor_products = competitor_reference_df['sku'].unique()
    
    months = sorted(reference_df['year_month'].unique())
    num_months = len(months)
    num_own = len(own_products)
    num_comp = len(competitor_products)
    
    own_index = {p: i for i, p in enumerate(own_products)}
    competitor_index = {p: i for i, p in enumerate(competitor_products)}
    
    E_price_to_volume = np.zeros((num_own, num_own))
    
    for _, row in own_to_own_elasticity_df.iterrows():
        i = own_index[row['target_sku']]
        j = own_index[row['other_sku']]
        E_price_to_volume[i, j] = row['elasticity']
    
    E_price_to_comp_volume = np.zeros((num_own, num_comp))
    
    for _, row in own_to_competitor_elasticity_df.iterrows():
        i = own_index[row['target_sku']]
        j = competitor_index[row['other_sku']]
        E_price_to_comp_volume[i, j] = row['elasticity']

    return own_products, competitor_products, months, num_months, num_own, num_comp, E_price_to_volume, E_price_to_comp_volume


def create_reference_arrays(reference_df, competitor_reference_df, months, own_products, competitor_products):
    """
    Create numpy arrays from reference dataframes for efficient computation.
    
    Args:
        reference_df: Processed reference data for own products
        competitor_reference_df: Processed reference data for competitor products
        months: List of time periods
        own_products: List of own product SKUs
        competitor_products: List of competitor product SKUs
    
    Returns:
        Dictionary containing all reference arrays
    """
    M = len(months)
    N = len(own_products)
    K = len(competitor_products)

    month_to_idx = {m: i for i, m in enumerate(months)}
    own_to_idx = {s: i for i, s in enumerate(own_products)}
    comp_to_idx = {s: i for i, s in enumerate(competitor_products)}

    ref_price_liter = np.zeros((M, N), dtype=float)
    ref_price_unit = np.zeros((M, N), dtype=float)
    ref_vol_own_sellin = np.zeros((M, N), dtype=float)
    ref_vol_own_sellout = np.zeros((M, N), dtype=float)
    capacity = np.zeros((M, N), dtype=float)
    markup = np.zeros((M, N), dtype=float)
    discount = np.zeros((M, N), dtype=float)
    excise = np.zeros((M, N), dtype=float)
    vilc = np.zeros((M, N), dtype=float)

    for _, row in reference_df.iterrows():
        mi = month_to_idx[row['year_month']]
        si = own_to_idx[row['sku']]
        ref_price_liter[mi, si] = row['reference_price']
        ref_price_unit[mi, si] = row['reference_price_unit']
        ref_vol_own_sellin[mi, si] = row['reference_volume']
        ref_vol_own_sellout[mi, si] = row['sellout_volume']
        capacity[mi, si] = row['capacity']
        markup[mi, si] = row['markup']
        discount[mi, si] = row['discount']
        excise[mi, si] = row['excise']
        vilc[mi, si] = row['vilc']

    ref_price_comp = np.zeros((M, K), dtype=float)
    ref_vol_comp = np.zeros((M, K), dtype=float)
    for _, row in competitor_reference_df.iterrows():
        mi = month_to_idx[row['year_month']]
        ci = comp_to_idx[row['sku']]
        ref_price_comp[mi, ci] = row['reference_price']
        ref_vol_comp[mi, ci] = row['reference_volume']

    return {
        'ref_price_liter': ref_price_liter,
        'ref_price_unit': ref_price_unit,
        'ref_vol_own_sellin': ref_vol_own_sellin,
        'ref_vol_own_sellout': ref_vol_own_sellout,
        'capacity': capacity,
        'markup': markup,
        'discount': discount,
        'excise': excise,
        'vilc': vilc,
        'ref_price_comp': ref_price_comp,
        'ref_vol_comp': ref_vol_comp,
        'M': M,
        'N': N,
        'K': K
    }


def create_bounds(reference_df_padded_bound, sku_lower_bound, sku_upper_bound):
    """
    Create price bounds for optimization variables.
    
    Args:
        reference_df_padded_bound: Padded reference dataframe with presence indicators
        sku_lower_bound: Lower bound for price changes
        sku_upper_bound: Upper bound for price changes
    
    Returns:
        List of (lower, upper) bound tuples for each optimization variable
    """
    reference_df_padded_bound['reference_price_unit'] = (
        reference_df_padded_bound['reference_price'] * reference_df_padded_bound['capacity'] / 1000
    )

    ref_prices = reference_df_padded_bound['reference_price_unit'].values
    is_present = reference_df_padded_bound['present'] == 'present'
    ref_prices_filled = np.where(np.isnan(ref_prices), 0, ref_prices)

    bounds = []
    for present, ref_price in zip(is_present, ref_prices_filled):
        if present:
            lower = max(ref_price + sku_lower_bound, 0.0001)
            upper = ref_price + sku_upper_bound
            bounds.append((lower, upper))
        else:
            bounds.append((ref_price, ref_price))
    
    return bounds


def create_segment_size_labels(reference_df, num_months, num_own):
    """
    Create segment and size labels arrays for hierarchy constraints.
    
    Args:
        reference_df: Processed reference dataframe with segment and size_group columns
        num_months: Number of months (M)
        num_own: Number of own products (N)
    
    Returns:
        Tuple of (segment_labels, size_labels) as numpy arrays of shape (M, N)
    """
    segment_labels = np.array(reference_df["segment"])
    segment_labels = segment_labels.reshape(num_months, num_own)
    
    size_labels = np.array(reference_df["size_group"])
    size_labels = size_labels.reshape(num_months, num_own)
    
    return segment_labels, size_labels
