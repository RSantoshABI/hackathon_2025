"""
Optimization module for price optimization.
Optimized with pre-computed arrays and conditional parallelization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
from joblib import Parallel, delayed


class PriceOptimizer:
    """Handles price optimization efficiently."""

    def __init__(
        self,
        reference_df,
        competitor_reference_df,
        own_products,
        competitor_products,
        months,
        num_months,
        num_own,
        num_comp,
        E_price_to_volume,
        E_price_to_comp_volume,
        VAT=0.19
    ):
        """Initialize optimizer with flexible parameters."""
        self.reference_df = reference_df
        self.competitor_reference_df = competitor_reference_df
        self.own_products = own_products
        self.competitor_products = competitor_products
        self.months = months
        self.num_months = num_months
        self.num_own = num_own
        self.num_comp = num_comp
        self.E_price_to_volume = E_price_to_volume
        self.E_price_to_comp_volume = E_price_to_comp_volume
        self.VAT = VAT
        self.monthly_outputs_df = None

        # Conditional parallelization: enable only for larger datasets
        self.use_parallel = num_own > 15
        self.n_jobs = -1 if self.use_parallel else 1

        # Pre-compute reference arrays and mappings for performance
        print("Pre-computing reference arrays...")
        self._precompute_reference_arrays()
        self._precompute_segment_mappings()
        self._precompute_reference_prices()
        print("Pre-computation complete!")

        # Report parallelization strategy
        if self.use_parallel:
            print(f"✅ Parallelization ENABLED: {num_own} SKUs > 15 threshold")
            print(f"   Using {self.n_jobs} parallel jobs for optimization")
        else:
            print(f"⚡ Sequential mode: {num_own} SKUs <= 15 threshold")
            print("   Sequential processing is faster for small datasets")

    def _precompute_reference_arrays(self):
        """Pre-compute reference arrays to avoid DataFrame filtering in objective."""
        self.ref_arrays_own = {}
        self.ref_arrays_comp = {}

        for month in self.months:
            # Pre-filter and convert to numpy arrays once for own products
            df_own = self.reference_df[
                (self.reference_df['year_month'] == month) &
                (self.reference_df['sku'].isin(self.own_products))
            ].set_index('sku').reindex(self.own_products)

            self.ref_arrays_own[month] = {
                'volume': df_own['reference_volume'].values,
                'volume_sellout': df_own['sellout_volume'].values,
                'price': df_own['reference_price'].values,
                'capacity': df_own['capacity'].values,
                'markup': df_own['markup'].values,
                'discount': df_own['discount'].values,
                'excise': df_own['excise'].values,
                'vilc': df_own['vilc'].values
            }

            # Pre-filter and convert to numpy arrays once for competitor products
            df_comp = self.competitor_reference_df[
                (self.competitor_reference_df['year_month'] == month) &
                (self.competitor_reference_df['sku'].isin(self.competitor_products))
            ].set_index('sku').reindex(self.competitor_products)

            self.ref_arrays_comp[month] = {
                'volume': df_comp['reference_volume'].values,
                'price': df_comp['reference_price'].values
            }

    def _precompute_segment_mappings(self):
        """Pre-compute segment and size group mappings as numpy arrays."""
        # Create first occurrence mapping for own products
        first_occurrence = self.reference_df[
            self.reference_df['sku'].isin(self.own_products)
        ].drop_duplicates(subset='sku', keep='first').set_index('sku')

        # Get segment and size for each product in order
        segment_list = []
        size_list = []
        for sku in self.own_products:
            if sku in first_occurrence.index:
                segment_list.append(first_occurrence.loc[sku, 'segment'])
                size_list.append(first_occurrence.loc[sku, 'size_group'])
            else:
                segment_list.append('Unknown')
                size_list.append('Unknown')

        # Create mappings
        self.segment_map = {'Value': 0, 'Core': 1, 'Core+': 2,
                            'Premium': 3, 'Super Premium': 4, 'Unknown': -1}
        self.size_map = {'Small': 0, 'Regular': 1, 'Large': 2, 'Unknown': -1}

        self.segment_indices = np.array([
            self.segment_map.get(seg, -1) for seg in segment_list
        ])
        self.size_indices = np.array([
            self.size_map.get(sz, -1) for sz in size_list
        ])

    def _precompute_reference_prices(self):
        """Pre-compute flattened reference prices for penalty calculation."""
        ref_prices_unit = (
            self.reference_df['reference_price'] *
            self.reference_df['capacity'] / 1000
        )
        self.ref_prices_unit_flat = ref_prices_unit.values

    def get_reference_arrays_own(self, month, ref_df, product_list):
        """Get reference arrays for own products for a specific month."""
        df_m = ref_df[
            (ref_df['year_month'] == month) &
            (ref_df['sku'].isin(product_list))
        ].set_index('sku').reindex(product_list)

        volume = df_m['reference_volume'].values
        volume_sellout = df_m['sellout_volume'].values
        price = df_m['reference_price'].values
        capacity = df_m['capacity'].values
        markup = df_m['markup'].values
        discount = df_m['discount'].values
        excise = df_m['excise'].values
        vilc = df_m['vilc'].values

        return (volume, volume_sellout, price, capacity,
                markup, discount, excise, vilc)

    def get_reference_arrays_comp(self, month, ref_df, product_list):
        """Get reference arrays for competitor products for a month."""
        df_m = ref_df[
            (ref_df['year_month'] == month) &
            (ref_df['sku'].isin(product_list))
        ].set_index('sku').reindex(product_list)

        volume = df_m['reference_volume'].values
        price = df_m['reference_price'].values

        return volume, price

    def calc_volume(
        self,
        opt_price_unit,
        ref_volume_own,
        ref_volume_sellout,
        ref_volume_comp,
        ref_price_liter,
        capacity,
        elasticity_matrix_own,
        elasticity_matrix_comp
    ):
        """Calculate volume given price changes and elasticities."""
        opt_price_liter = opt_price_unit * 1000 / capacity
        price_ratio_own = np.log(opt_price_liter / ref_price_liter)

        elasticity_effect = elasticity_matrix_own.T @ price_ratio_own
        Q_own_opt = ref_volume_own * np.exp(elasticity_effect)
        Q_own_opt_sellout = ref_volume_sellout * np.exp(elasticity_effect)
        Q_comp_opt = ref_volume_comp * np.exp(
            elasticity_matrix_comp.T @ price_ratio_own
        )

        return Q_own_opt, Q_comp_opt, Q_own_opt_sellout

    def calc_MACO(
        self,
        opt_price_unit,
        opt_volume,
        capacity,
        markup,
        discount,
        excise,
        vilc
    ):
        """Calculate MACO and NR based on volume and costs."""
        sales_units = (opt_volume * 100000) / capacity
        discount_pct = discount
        excise_pct = excise
        NR_per_unit = (
            (opt_price_unit / (1 + markup)) *
            (1 + discount_pct + excise_pct)
        ) / (1 + self.VAT)
        NR = NR_per_unit * sales_units
        MACO = NR + vilc

        return NR, MACO, sales_units

    def multiples_of_50_penalty(self, P_opt, P_ref, penalty_weight=1.0):
        """Soft constraint - penalty for prices not multiples of 50."""
        diff = P_opt - P_ref
        remainder = np.mod(diff, 50)
        penalty_per_sku = np.minimum(remainder**2, (50 - remainder)**2)
        total_penalty = penalty_weight * np.sum(penalty_per_sku)
        return total_penalty

    def round_to_nearest_50(self, price_array, ref_price_array):
        """Round price differences to nearest 50."""
        price_diff = price_array - ref_price_array
        price_diff_rounded = 50 * np.round(price_diff / 50)
        return ref_price_array + price_diff_rounded

    def _process_month_for_objective(self, i, month, price_opt_own):
        """
        Process a single month for objective calculation.
        Used for parallel execution when num_own > 15.
        Returns: (MACO_sum, volumes, NR_values)
        """
        # Use pre-computed arrays
        ref = self.ref_arrays_own[month]
        comp = self.ref_arrays_comp[month]

        # Fast numpy calculations
        vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
            price_opt_own, ref['volume'], ref['volume_sellout'],
            comp['volume'], ref['price'], ref['capacity'],
            self.E_price_to_volume, self.E_price_to_comp_volume
        )

        NR_own, MACO_own, _ = self.calc_MACO(
            price_opt_own, vol_own_opt, ref['capacity'],
            ref['markup'], ref['discount'], ref['excise'], ref['vilc']
        )

        return MACO_own.sum(), vol_own_opt, NR_own

    def objective(self, P_opt):
        """
        Objective function - optimized with conditional parallelization.
        Maximizes MACO with hierarchy penalties.
        Uses pre-computed arrays and parallel processing for large datasets.
        """
        prices_by_month = P_opt.reshape(self.num_months, self.num_own)

        # Pre-allocate numpy arrays for all data
        total_size = self.num_months * self.num_own
        volumes_all = np.zeros(total_size)
        NR_all = np.zeros(total_size)
        segment_indices_all = np.zeros(total_size, dtype=int)
        size_indices_all = np.zeros(total_size, dtype=int)

        if self.use_parallel:
            # PARALLEL MODE: Process months in parallel (for large datasets)
            results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(self._process_month_for_objective)(
                    i, month, prices_by_month[i, :]
                )
                for i, month in enumerate(self.months)
            )

            # Aggregate results from parallel processing
            total_MACO = 0
            for i, (maco_sum, vol_own_opt, NR_own) in enumerate(results):
                total_MACO += maco_sum
                start_idx = i * self.num_own
                end_idx = (i + 1) * self.num_own
                volumes_all[start_idx:end_idx] = vol_own_opt
                NR_all[start_idx:end_idx] = NR_own
                segment_indices_all[start_idx:end_idx] = self.segment_indices
                size_indices_all[start_idx:end_idx] = self.size_indices
        else:
            # SEQUENTIAL MODE: Process months sequentially (for small datasets)
            total_MACO = 0
            for i, month in enumerate(self.months):
                maco_sum, vol_own_opt, NR_own = \
                    self._process_month_for_objective(
                        i, month, prices_by_month[i, :]
                    )

                total_MACO += maco_sum
                start_idx = i * self.num_own
                end_idx = (i + 1) * self.num_own
                volumes_all[start_idx:end_idx] = vol_own_opt
                NR_all[start_idx:end_idx] = NR_own
                segment_indices_all[start_idx:end_idx] = self.segment_indices
                size_indices_all[start_idx:end_idx] = self.size_indices

        # Calculate penalties using fast numpy operations
        segment_penalty = self._calculate_segment_penalty_numpy(
            volumes_all, NR_all, segment_indices_all
        )
        size_penalty = self._calculate_size_penalty_numpy(
            volumes_all, NR_all, size_indices_all
        )

        # Use pre-computed reference prices
        mult50_penalty = self.multiples_of_50_penalty(
            P_opt, self.ref_prices_unit_flat, 1e5
        )

        return -total_MACO + segment_penalty + size_penalty + mult50_penalty

    def _calculate_segment_penalty_numpy(self, volumes, NR_values, segment_indices):
        """
        Fast numpy-based segment penalty calculation.
        Replaces pandas groupby operations.
        """
        penalty = 0.0

        # Iterate through segment pairs (Value->Core, Core->Core+, etc.)
        for seg_idx in range(4):  # 0-3 (Value to Premium)
            mask_a = segment_indices == seg_idx
            mask_b = segment_indices == (seg_idx + 1)

            if mask_a.any() and mask_b.any():
                vol_a = volumes[mask_a].sum()
                vol_b = volumes[mask_b].sum()

                if vol_a > 0 and vol_b > 0:
                    nr_per_hl_a = NR_values[mask_a].sum() / vol_a
                    nr_per_hl_b = NR_values[mask_b].sum() / vol_b

                    diff = nr_per_hl_a - nr_per_hl_b
                    if diff > 0:
                        penalty += diff * 1e5

        return penalty

    def _calculate_segment_penalty(self):
        """Calculate penalty for segment hierarchy violations (legacy)."""
        segment_group = self.monthly_outputs_df.groupby('segment').agg({
            'NR_opt': 'sum',
            'volume_opt': 'sum'
        })
        segment_group['NR_per_HL'] = (
            segment_group['NR_opt'] / segment_group['volume_opt']
        )

        segment_order = ['Value', 'Core', 'Core+', 'Premium', 'Super Premium']
        segment_penalty = 0

        for i in range(len(segment_order) - 1):
            seg_a = segment_order[i]
            seg_b = segment_order[i + 1]
            if seg_a in segment_group.index and seg_b in segment_group.index:
                diff = (
                    segment_group.loc[seg_a, 'NR_per_HL'] -
                    segment_group.loc[seg_b, 'NR_per_HL']
                )
                if diff > 0:
                    segment_penalty += diff * 1e5

        return segment_penalty

    def _calculate_size_penalty_numpy(self, volumes, NR_values, size_indices):
        """
        Fast numpy-based size penalty calculation.
        Replaces pandas groupby operations.
        """
        penalty = 0.0

        # Iterate through size pairs (Small->Regular, Regular->Large)
        for sz_idx in range(2):  # 0-1 (Small to Regular, Regular to Large)
            mask_a = size_indices == sz_idx
            mask_b = size_indices == (sz_idx + 1)

            if mask_a.any() and mask_b.any():
                vol_a = volumes[mask_a].sum()
                vol_b = volumes[mask_b].sum()

                if vol_a > 0 and vol_b > 0:
                    nr_per_hl_a = NR_values[mask_a].sum() / vol_a
                    nr_per_hl_b = NR_values[mask_b].sum() / vol_b

                    # For size: larger should have higher NR per HL
                    diff = nr_per_hl_b - nr_per_hl_a
                    if diff > 0:
                        penalty += diff * 1e5

        return penalty

    def _calculate_size_penalty(self):
        """Calculate penalty for size group hierarchy violations (legacy)."""
        size_group = self.monthly_outputs_df.groupby('size_group').agg({
            'NR_opt': 'sum',
            'volume_opt': 'sum'
        })
        size_group['NR_per_HL'] = (
            size_group['NR_opt'] / size_group['volume_opt']
        )

        size_order = ['Small', 'Regular', 'Large']
        size_penalty = 0

        for i in range(len(size_order) - 1):
            sz_a = size_order[i]
            sz_b = size_order[i + 1]
            if sz_a in size_group.index and sz_b in size_group.index:
                diff = (
                    size_group.loc[sz_b, 'NR_per_HL'] -
                    size_group.loc[sz_a, 'NR_per_HL']
                )
                if diff > 0:
                    size_penalty += diff * 1e5

        return size_penalty

    def create_bounds(self, reference_df_padded_bound):
        """Create price bounds for optimization."""
        reference_df_padded_bound['reference_price_unit'] = (
            reference_df_padded_bound['reference_price'] *
            reference_df_padded_bound['capacity'] / 1000
        )

        ref_prices = reference_df_padded_bound['reference_price_unit'].values
        is_present = reference_df_padded_bound['present'] == 'present'
        ref_prices_filled = np.where(np.isnan(ref_prices), 0, ref_prices)

        bounds = []
        for present, ref_price in zip(is_present, ref_prices_filled):
            if present:
                lower = max(ref_price - 300, 0)
                upper = ref_price + 500
                bounds.append((lower, upper))
            else:
                bounds.append((ref_price, ref_price))

        return bounds

    def get_industry_volumes(self, P_opt):
        """Calculate industry volumes for all manufacturers (optimized)."""
        prices_by_month = P_opt.reshape(self.num_months, self.num_own)

        abi_sku_all = []
        abi_year_month_all = []
        abi_volume_ref_all = []
        abi_volume_opt_all = []
        comp_sku_all = []
        comp_year_month_all = []
        comp_volume_ref_all = []
        comp_volume_opt_all = []

        for i, month in enumerate(self.months):
            # Use pre-computed arrays instead of DataFrame filtering
            ref = self.ref_arrays_own[month]
            comp = self.ref_arrays_comp[month]

            price_opt_own = prices_by_month[i, :]

            vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
                price_opt_own, ref['volume'], ref['volume_sellout'],
                comp['volume'], ref['price'], ref['capacity'],
                self.E_price_to_volume, self.E_price_to_comp_volume
            )

            abi_sku_all.extend(self.own_products)
            abi_year_month_all.extend([month] * len(self.own_products))
            abi_volume_ref_all.extend(ref['volume_sellout'])
            abi_volume_opt_all.extend(vol_own_opt_sellout)
            comp_sku_all.extend(self.competitor_products)
            comp_year_month_all.extend([month] * len(self.competitor_products))
            comp_volume_ref_all.extend(comp['volume'])
            comp_volume_opt_all.extend(vol_comp_opt)

        abi_vol_df = pd.DataFrame({
            'sku': abi_sku_all,
            'manufacturer': 'abi',
            'year_month': abi_year_month_all,
            'volume_ref': abi_volume_ref_all,
            'volume_opt': abi_volume_opt_all,
        })
        comp_vol_df = pd.DataFrame({
            'sku': comp_sku_all,
            'manufacturer': 'competitor',
            'year_month': comp_year_month_all,
            'volume_ref': comp_volume_ref_all,
            'volume_opt': comp_volume_opt_all,
        })

        return pd.concat([abi_vol_df, comp_vol_df], ignore_index=True)

    def generate_monthly_outputs_df(self, P_opt):
        """
        Generate detailed monthly outputs DataFrame.
        Called after optimization to get full results.
        """
        prices_by_month = P_opt.reshape(self.num_months, self.num_own)

        sku_all = []
        year_month_all = []
        price_liter_ref_all = []
        price_unit_ref_all = []
        volume_ref_all = []
        volume_ref_sellout_all = []
        NR_ref_all = []
        MACO_ref_all = []
        price_liter_opt_all = []
        price_unit_opt_all = []
        volume_opt_all = []
        volume_opt_sellout_all = []
        NR_opt_all = []
        MACO_opt_all = []

        for i, month in enumerate(self.months):
            # Use pre-computed arrays
            ref = self.ref_arrays_own[month]

            price_opt_own = prices_by_month[i, :]
            price_opt_liter = price_opt_own * 1000 / ref['capacity']
            price_ref_unit = ref['price'] * ref['capacity'] / 1000

            vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
                price_opt_own, ref['volume'], ref['volume_sellout'],
                self.ref_arrays_comp[month]['volume'], ref['price'],
                ref['capacity'], self.E_price_to_volume,
                self.E_price_to_comp_volume
            )

            NR_own, MACO_own, _ = self.calc_MACO(
                price_opt_own, vol_own_opt, ref['capacity'],
                ref['markup'], ref['discount'], ref['excise'], ref['vilc']
            )

            NR_ref, MACO_ref, _ = self.calc_MACO(
                price_ref_unit, ref['volume'], ref['capacity'],
                ref['markup'], ref['discount'], ref['excise'], ref['vilc']
            )

            sku_all.extend(self.own_products)
            year_month_all.extend([month] * len(self.own_products))
            price_liter_ref_all.extend(ref['price'])
            price_unit_ref_all.extend(price_ref_unit)
            volume_ref_all.extend(ref['volume'])
            volume_ref_sellout_all.extend(ref['volume_sellout'])
            NR_ref_all.extend(NR_ref)
            MACO_ref_all.extend(MACO_ref)
            price_liter_opt_all.extend(price_opt_liter)
            price_unit_opt_all.extend(price_opt_own)
            volume_opt_all.extend(vol_own_opt)
            volume_opt_sellout_all.extend(vol_own_opt_sellout)
            NR_opt_all.extend(NR_own)
            MACO_opt_all.extend(MACO_own)

        self.monthly_outputs_df = pd.DataFrame({
            'sku': sku_all,
            'year_month': year_month_all,
            'price_liter_ref': price_liter_ref_all,
            'price_unit_ref': price_unit_ref_all,
            'volume_ref': volume_ref_all,
            'volume_ref_sellout': volume_ref_sellout_all,
            'NR_ref': NR_ref_all,
            'MACO_ref': MACO_ref_all,
            'price_liter_opt': price_liter_opt_all,
            'price_unit_opt': price_unit_opt_all,
            'volume_opt': volume_opt_all,
            'volume_opt_sellout': volume_opt_sellout_all,
            'NR_opt': NR_opt_all,
            'MACO_opt': MACO_opt_all
        })

        self.monthly_outputs_df = self.monthly_outputs_df.merge(
            self.reference_df[[
                'sku', 'year_month', 'segment', 'size_group', 'capacity'
            ]],
            on=['sku', 'year_month'],
            how='left'
        )
        self.monthly_outputs_df['HL'] = self.monthly_outputs_df['volume_opt']

        return self.monthly_outputs_df

    def optimize(
        self,
        constraints,
        bounds,
        P0,
        method='trust-constr',
        options=None
    ):
        """Run optimization."""
        if options is None:
            options = {'disp': True}

        print("Starting optimization...")
        start = time.time()

        result = minimize(
            self.objective,
            P0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options
        )

        elapsed = time.time() - start
        print(f"Optimization took {elapsed:.4f} seconds")

        # Generate detailed outputs after optimization
        monthly_outputs_df_unrounded = self.generate_monthly_outputs_df(result.x)
        opt_industry_volume_df = self.get_industry_volumes(result.x)

        rounded_prices = self.round_to_nearest_50(result.x, P0)
        monthly_outputs_df_rounded = self.generate_monthly_outputs_df(rounded_prices)
        opt_vol_rounded = self.get_industry_volumes(rounded_prices)

        return (
            result,
            monthly_outputs_df_unrounded,
            opt_industry_volume_df,
            rounded_prices,
            monthly_outputs_df_rounded,
            opt_vol_rounded
        )
