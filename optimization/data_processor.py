"""
Data processing module for price optimization.
Handles data loading, cleaning, preprocessing, and feature engineering.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List


class DataProcessor:
    """Processes and prepares data for price optimization."""

    def __init__(
        self,
        min_period: str = '2024-01',
        max_period: str = '2026-12',
        vilc_gr: float = 0.0378
    ):
        """
        Initialize DataProcessor.

        Args:
            min_period: Minimum allowed period (YYYY-MM format)
            max_period: Maximum allowed period (YYYY-MM format)
            vilc_gr: VILC annual growth rate
        """
        self.min_period = min_period
        self.max_period = max_period
        self.vilc_gr = vilc_gr

    def generate_periods(
        self, start_period: str, end_period: str
    ) -> List[str]:
        """
        Generate list of periods between start and end dates.

        Args:
            start_period: Start period in YYYY-MM format
            end_period: End period in YYYY-MM format

        Returns:
            List of period strings in YYYY-MM format
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

    def adjust_period(
        self, period: str, years_to_subtract: int
    ) -> str:
        """
        Adjust period by subtracting years.

        Args:
            period: Period in YYYY-MM format
            years_to_subtract: Number of years to subtract

        Returns:
            Adjusted period string in YYYY-MM format
        """
        period_date = datetime.strptime(period, "%Y-%m")
        adjusted_date = period_date.replace(
            year=period_date.year - years_to_subtract
        )
        return adjusted_date.strftime("%Y-%m")

    def check_periods_in_valid_range(
        self, periods: List[str], valid_periods: set
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Check if periods are valid, falling back to prior years if needed.

        Args:
            periods: List of periods to check
            valid_periods: Set of valid periods from the data

        Returns:
            Tuple of (adjusted_periods, period_mapping) where
            period_mapping maps original to adjusted periods
        """
        adjusted_periods = []
        period_mapping = {}
        for period in periods:
            original_period = period
            if period in valid_periods:
                adjusted_periods.append(period)
                period_mapping[original_period] = period
            else:
                adjusted_period_1 = self.adjust_period(period, 1)
                if adjusted_period_1 in valid_periods:
                    adjusted_periods.append(adjusted_period_1)
                    period_mapping[original_period] = adjusted_period_1
                else:
                    adjusted_period_2 = self.adjust_period(
                        adjusted_period_1, 1
                    )
                    if adjusted_period_2 in valid_periods:
                        adjusted_periods.append(adjusted_period_2)
                        period_mapping[original_period] = adjusted_period_2
                    else:
                        msg = (
                            f"Period {period} is invalid - "
                            "no valid data within 2 years prior"
                        )
                        raise ValueError(msg)
        return adjusted_periods, period_mapping

    def filter_dataframe(
        self, df: pd.DataFrame, start_period: str, end_period: str
    ) -> pd.DataFrame:
        """
        Filter dataframe by period range with intelligent fallback
        to prior years.

        Args:
            df: DataFrame with year_month column
            start_period: Start period in YYYY-MM format
            end_period: End period in YYYY-MM format

        Returns:
            Filtered DataFrame with used_prior flag indicating which
            rows used fallback data
        """
        valid_periods = set(df['year_month'].unique())

        # Generate all periods in the requested range
        periods_to_check = self.generate_periods(start_period, end_period)

        # Check which periods are valid and create mapping
        adjusted_periods, period_mapping = (
            self.check_periods_in_valid_range(
                periods_to_check, valid_periods
            )
        )

        # Filter dataframe to only include adjusted periods
        filtered_df = df[df['year_month'].isin(adjusted_periods)].copy()

        # Create reverse mapping to track which periods used fallback
        period_map_rev = {v: k for k, v in period_mapping.items()}
        filtered_df['year_month_map'] = (
            filtered_df['year_month'].map(period_map_rev)
        )

        # Mark rows that used prior year data
        filtered_df['used_prior'] = (
            filtered_df['year_month'] != filtered_df['year_month_map']
        )

        # Rename columns to keep the mapped period as main year_month
        filtered_df.rename(
            columns={'year_month': 'year_month_og'}, inplace=True
        )
        filtered_df.rename(
            columns={'year_month_map': 'year_month'}, inplace=True
        )
        filtered_df.drop(columns=['year_month_og'], inplace=True)

        # Reorder columns to put year_month first
        cols = list(filtered_df.columns)
        if 'year_month' in cols and 'used_prior' in cols:
            cols.remove('year_month')
            cols.remove('used_prior')
            cols = ['year_month'] + cols + ['used_prior']
            filtered_df = filtered_df[cols]

        # Sort by year_month and sku for consistency
        filtered_df = filtered_df.sort_values(by=['year_month', 'sku'])

        return filtered_df

    def extract_pack_type(self, sku: str) -> str:
        """Extract pack type from SKU."""
        known_pack_types = ['NRB', 'RB', 'CAN']
        for pt in known_pack_types:
            if pt in sku:
                return pt
        return 'UNKNOWN'

    def assign_size_group(self, row: pd.Series) -> str:
        """Assign size group based on capacity and pack type."""
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

    def pad_reference_dataframe(
        self,
        reference_df: pd.DataFrame,
        own_skus: np.ndarray,
        all_months: np.ndarray
    ) -> pd.DataFrame:
        """Pad reference dataframe with missing SKU-month combinations."""
        all_months_date = pd.to_datetime(all_months)
        all_months_date = pd.Series(all_months_date).sort_values().unique()

        full_index_own = pd.MultiIndex.from_product(
            [all_months_date, own_skus],
            names=['year_month_date', 'sku']
        )

        reference_df['year_month_date'] = pd.to_datetime(
            reference_df['year_month']
        )
        reference_df = reference_df.set_index(['year_month_date', 'sku'])

        reference_df_padded = reference_df.reindex(full_index_own)

        reference_df_padded['present'] = 'present'
        mask = reference_df_padded['reference_volume'].isna()
        reference_df_padded.loc[mask, 'present'] = 'missing'

        reference_df_padded['reference_volume'] = (
            reference_df_padded['reference_volume'].fillna(0)
        )
        reference_df_padded['sellout_volume'] = (
            reference_df_padded['sellout_volume'].fillna(0)
        )

        mean_prices = reference_df['reference_price'].mean()
        reference_df_padded['reference_price'] = (
            reference_df_padded['reference_price'].fillna(mean_prices)
        )

        for col in ['markup', 'discount', 'excise', 'vilc']:
            if col in reference_df_padded.columns:
                reference_df_padded[col] = reference_df_padded[col].fillna(0)

        reference_df_padded = reference_df_padded.reset_index()

        year_month_mapping = reference_df.reset_index()
        year_month_mapping = year_month_mapping[
            ['year_month_date', 'year_month']
        ].drop_duplicates()

        reference_df_padded = pd.merge(
            reference_df_padded, year_month_mapping,
            how='left', on='year_month_date'
        )
        reference_df_padded = reference_df_padded.drop(
            columns={'year_month_date', 'year_month_x'}
        )
        reference_df_padded.rename(
            columns={'year_month_y': 'year_month'}, inplace=True
        )

        return reference_df_padded

    def pad_competitor_dataframe(
        self,
        competitor_df: pd.DataFrame,
        competitor_skus: np.ndarray,
        all_months: np.ndarray
    ) -> pd.DataFrame:
        """Pad competitor dataframe with missing SKU-month combinations."""
        all_months_date = pd.to_datetime(all_months)
        all_months_date = pd.Series(all_months_date).sort_values().unique()

        full_index = pd.MultiIndex.from_product(
            [all_months_date, competitor_skus],
            names=['year_month_date', 'sku']
        )

        competitor_df['year_month_date'] = pd.to_datetime(
            competitor_df['year_month']
        )
        competitor_df = competitor_df.set_index([
            'year_month_date', 'sku'
        ])

        comp_df_padded = competitor_df.reindex(full_index)

        comp_df_padded['reference_volume'] = (
            comp_df_padded['reference_volume'].fillna(0)
        )

        mean_prices = comp_df_padded['reference_price'].mean()
        comp_df_padded['reference_price'] = (
            comp_df_padded['reference_price'].fillna(mean_prices)
        )

        comp_df_padded = comp_df_padded.reset_index()

        year_month_mapping = competitor_df.reset_index()
        year_month_mapping = year_month_mapping[
            ['year_month_date', 'year_month']
        ].drop_duplicates()

        comp_df_padded = pd.merge(
            comp_df_padded, year_month_mapping,
            how='left', on='year_month_date'
        )
        comp_df_padded = comp_df_padded.drop(
            columns={'year_month_date', 'year_month_x'}
        )
        comp_df_padded.rename(
            columns={'year_month_y': 'year_month'}, inplace=True
        )

        return comp_df_padded

    def prepare_elasticity_matrix(
        self,
        own_to_own_elasticity_df: pd.DataFrame,
        own_index: Dict[str, int],
        num_own: int
    ) -> np.ndarray:
        """Prepare own-to-own elasticity matrix."""
        E_price_to_volume = np.zeros((num_own, num_own))

        for _, row in own_to_own_elasticity_df.iterrows():
            i = own_index[row['target_sku']]
            j = own_index[row['other_sku']]
            E_price_to_volume[i, j] = row['elasticity']

        return E_price_to_volume

    def prepare_competitor_elasticity_matrix(
        self,
        own_to_competitor_elasticity_df: pd.DataFrame,
        own_index: Dict[str, int],
        competitor_index: Dict[str, int],
        num_own: int,
        num_comp: int
    ) -> np.ndarray:
        """Prepare own-to-competitor elasticity matrix."""
        E_price_to_comp_volume = np.zeros((num_own, num_comp))

        for _, row in own_to_competitor_elasticity_df.iterrows():
            i = own_index[row['target_sku']]
            j = competitor_index[row['other_sku']]
            E_price_to_comp_volume[i, j] = row['elasticity']

        return E_price_to_comp_volume

    def load_and_process_data(
        self,
        elasticity_path: str,
        reference_path: str,
        competitor_elasticity_path: str,
        competitor_reference_path: str,
        seg_mapping_path: str,
        start_period: str,
        end_period: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict]:
        """
        Load and process all data files.

        Returns:
            Tuple of (reference_df, competitor_reference_df,
            E_own, E_comp, metadata)
        """
        # Load data
        own_to_own_elasticity_df = pd.read_csv(elasticity_path)
        reference_df = pd.read_csv(reference_path)
        own_to_competitor_elasticity_df = pd.read_csv(
            competitor_elasticity_path
        )
        competitor_reference_df = pd.read_csv(competitor_reference_path)
        seg_mapping = pd.read_csv(seg_mapping_path)

        print(
            f"Loaded data shapes: "
            f"{own_to_own_elasticity_df.shape}, "
            f"{reference_df.shape}, "
            f"{own_to_competitor_elasticity_df.shape}, "
            f"{competitor_reference_df.shape}"
        )

        # Filter by period
        reference_df_filt = self.filter_dataframe(
            reference_df, start_period, end_period
        )
        competitor_reference_df_filt = self.filter_dataframe(
            competitor_reference_df, start_period, end_period
        )

        # Apply VILC growth rate for fallback periods
        mask = reference_df_filt['used_prior']
        reference_df_filt.loc[mask, 'vilc'] = (
            reference_df_filt.loc[mask, 'vilc'] * (1 + self.vilc_gr)
        )

        reference_df = reference_df_filt.drop(columns={'used_prior'})
        competitor_reference_df = competitor_reference_df_filt.drop(
            columns={'used_prior'}
        )

        reference_df = reference_df.reset_index(drop=True)
        competitor_reference_df = competitor_reference_df.reset_index(
            drop=True
        )

        # Preprocess reference files
        own_products_init = reference_df['sku'].unique()
        competitor_products_init = competitor_reference_df['sku'].unique()

        mask1 = own_to_own_elasticity_df['target_sku'].isin(
            own_products_init
        )
        own_to_own_elasticity_df = own_to_own_elasticity_df[mask1]

        mask2 = own_to_own_elasticity_df['other_sku'].isin(
            own_products_init
        )
        own_to_own_elasticity_df = own_to_own_elasticity_df[mask2]

        mask3 = own_to_competitor_elasticity_df['target_sku'].isin(
            own_products_init
        )
        own_to_competitor_elasticity_df = (
            own_to_competitor_elasticity_df[mask3]
        )

        mask4 = own_to_competitor_elasticity_df['other_sku'].isin(
            competitor_products_init
        )
        own_to_competitor_elasticity_df = (
            own_to_competitor_elasticity_df[mask4]
        )

        reference_df = reference_df[[
            'year_month', 'sku', 'reference_volume', 'reference_price',
            'markup', 'discount', 'excise', 'vilc', 'sellout_volume'
        ]]

        # Extract unique SKUs
        own_skus = own_to_own_elasticity_df['target_sku'].unique()
        own_skus.sort()
        competitor_skus = own_to_competitor_elasticity_df['other_sku'].unique()
        competitor_skus.sort()

        all_months = reference_df['year_month'].unique()

        # Pad dataframes
        reference_df_padded = self.pad_reference_dataframe(
            reference_df, own_skus, all_months
        )
        reference_df_padded = pd.merge(
            reference_df_padded, seg_mapping, how='left', on='sku'
        )
        reference_df_padded_bound = reference_df_padded.copy()
        reference_df = reference_df_padded.drop(columns={'present'})

        all_months_comp = competitor_reference_df['year_month'].unique()
        competitor_reference_df = self.pad_competitor_dataframe(
            competitor_reference_df, competitor_skus, all_months_comp
        )

        # Add pack type and size group
        reference_df['pack_type'] = reference_df['sellin_sku'].apply(
            self.extract_pack_type
        )
        reference_df.loc[
            (reference_df['pack_type'] == 'UNKNOWN') &
            (reference_df['sku'].str.contains('NO RETORNABLE')),
            'pack_type'
        ] = 'NRB'
        reference_df.loc[
            (reference_df['pack_type'] == 'UNKNOWN') &
            (reference_df['sku'].str.contains('RETORNABLE')),
            'pack_type'
        ] = 'RB'

        reference_df['size_group'] = reference_df.apply(
            self.assign_size_group, axis=1
        )

        # Prepare elasticity matrices
        own_products = reference_df['sku'].unique()
        competitor_products = competitor_reference_df['sku'].unique()

        months = sorted(reference_df['year_month'].unique())
        num_months = len(months)
        num_own = len(own_products)
        num_comp = len(competitor_products)

        own_index = {p: i for i, p in enumerate(own_products)}
        competitor_index = {p: i for i, p in enumerate(competitor_products)}

        E_price_to_volume = self.prepare_elasticity_matrix(
            own_to_own_elasticity_df, own_index, num_own
        )
        E_price_to_comp_volume = (
            self.prepare_competitor_elasticity_matrix(
                own_to_competitor_elasticity_df,
                own_index,
                competitor_index,
                num_own,
                num_comp
            )
        )

        metadata = {
            'own_products': own_products,
            'competitor_products': competitor_products,
            'months': months,
            'num_months': num_months,
            'num_own': num_own,
            'num_comp': num_comp,
            'own_index': own_index,
            'competitor_index': competitor_index,
            'reference_df_padded_bound': reference_df_padded_bound
        }

        return (
            reference_df,
            competitor_reference_df,
            E_price_to_volume,
            E_price_to_comp_volume,
            metadata
        )
