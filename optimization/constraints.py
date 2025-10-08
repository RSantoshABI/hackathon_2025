"""
Constraints module for price optimization.
Defines all optimization constraints.
"""

import numpy as np
from scipy.optimize import NonlinearConstraint


class ConstraintManager:
    """Manages optimization constraints."""

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
        target_delta,
        VAT,
        get_reference_arrays_own_func,
        get_reference_arrays_comp_func,
        calc_volume_func,
        calc_MACO_func
    ):
        """
        Initialize constraint manager with all necessary data and functions.
        """
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
        self.target_delta = target_delta
        self.VAT = VAT

        # Store functions
        self.get_reference_arrays_own = get_reference_arrays_own_func
        self.get_reference_arrays_comp = get_reference_arrays_comp_func
        self.calc_volume = calc_volume_func
        self.calc_MACO = calc_MACO_func

        # Precompute reference values
        self.total_ref_industry_volume = (
            np.sum(reference_df['sellout_volume'].values) +
            np.sum(competitor_reference_df['reference_volume'].values)
        )
        self.total_ref_own_volume = np.sum(
            reference_df['reference_volume'].values)

        self.ref_market_share = (
            np.sum(reference_df['sellout_volume'].values) /
            self.total_ref_industry_volume
        )
        self.weighted_avg_ref_price = (
            np.sum(reference_df['reference_volume'].values *
                   reference_df['reference_price'].values) /
            np.sum(reference_df['reference_volume'].values)
        )

    def industry_volume_constraint_monthly(
            self,
            P_opt_unit: np.ndarray
            ) -> float:
        """
        Industry volume constraint - ensures total industry volume
        doesn't decrease by more than threshold.
        """
        prices_by_month = P_opt_unit.reshape(self.num_months, self.num_own)

        total_industry_volume_opt = 0

        for i, month in enumerate(self.months):
            (ref_vol_own, ref_vol_sellout, ref_price_own, capacity_own,
             markup_own, discount_own, excise_own, vilc_own) = \
                self.get_reference_arrays_own(
                    month, self.reference_df, self.own_products
                )

            (ref_vol_comp, ref_price_comp) = \
                self.get_reference_arrays_comp(
                    month,
                    self.competitor_reference_df,
                    self.competitor_products
                )

            price_opt_own = prices_by_month[i, :]

            vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
                price_opt_own, ref_vol_own, ref_vol_sellout, ref_vol_comp,
                ref_price_own, capacity_own, self.E_price_to_volume,
                self.E_price_to_comp_volume
            )

            industry_vol_opt = vol_own_opt_sellout.sum() + vol_comp_opt.sum()
            total_industry_volume_opt += industry_vol_opt

        return total_industry_volume_opt

    def own_volume_constraint_monthly(self, P_opt_unit: np.ndarray) -> float:
        """
        ABI volume constraint - ensures own volume stays within bounds.
        """
        prices_by_month = P_opt_unit.reshape(self.num_months, self.num_own)

        total_own_volume_opt = 0

        for i, month in enumerate(self.months):
            (ref_vol_own, ref_vol_sellout, ref_price_own, capacity_own,
             markup_own, discount_own, excise_own, vilc_own) = \
                self.get_reference_arrays_own(
                    month, self.reference_df, self.own_products
                )

            (ref_vol_comp, ref_price_comp) = \
                self.get_reference_arrays_comp(
                    month,
                    self.competitor_reference_df,
                    self.competitor_products
                )

            price_opt_own = prices_by_month[i, :]

            vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
                price_opt_own, ref_vol_own, ref_vol_sellout, ref_vol_comp,
                ref_price_own, capacity_own, self.E_price_to_volume,
                self.E_price_to_comp_volume
            )

            total_own_volume_opt += vol_own_opt.sum()

        return total_own_volume_opt

    def market_share_constraint_monthly(self, P_opt_unit: np.ndarray) -> float:
        """
        Market share constraint - ensures market share doesn't decrease
        by more than threshold.
        """
        prices_by_month = P_opt_unit.reshape(self.num_months, self.num_own)

        total_industry_volume_opt = 0
        total_own_volume_opt = 0

        for i, month in enumerate(self.months):
            (ref_vol_own, ref_vol_sellout, ref_price_own, capacity_own,
             markup_own, discount_own, excise_own, vilc_own) = \
                self.get_reference_arrays_own(
                    month, self.reference_df, self.own_products
                )

            (ref_vol_comp, ref_price_comp) = \
                self.get_reference_arrays_comp(
                    month, self.competitor_reference_df,
                    self.competitor_products
                )

            price_opt_own = prices_by_month[i, :]

            vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
                price_opt_own, ref_vol_own, ref_vol_sellout, ref_vol_comp,
                ref_price_own, capacity_own, self.E_price_to_volume,
                self.E_price_to_comp_volume
            )

            industry_vol_opt = vol_own_opt_sellout.sum() + vol_comp_opt.sum()
            total_industry_volume_opt += industry_vol_opt
            total_own_volume_opt += vol_own_opt_sellout.sum()

        ms_opt = total_own_volume_opt / total_industry_volume_opt
        return ms_opt

    def portfolio_pinc_constraint_monthly(
            self, P_opt_unit: np.ndarray
            ) -> float:
        """
        Portfolio PINC constraint - ensures portfolio price increase
        matches target.
        """
        prices_by_month = P_opt_unit.reshape(self.num_months, self.num_own)
        total_own_volume_opt = 0
        total_price_mult_vol_opt = 0

        for i, month in enumerate(self.months):
            (ref_vol_own, ref_vol_sellout, ref_price_own, capacity_own,
             markup_own, discount_own, excise_own, vilc_own) = \
                self.get_reference_arrays_own(
                    month, self.reference_df, self.own_products
                )

            (ref_vol_comp, ref_price_comp) = \
                self.get_reference_arrays_comp(
                    month,
                    self.competitor_reference_df,
                    self.competitor_products
                )

            price_opt_own = prices_by_month[i, :]
            price_opt_own_liter = price_opt_own * 100000 / capacity_own

            vol_own_opt, vol_comp_opt, vol_own_opt_sellout = self.calc_volume(
                price_opt_own, ref_vol_own, ref_vol_sellout, ref_vol_comp,
                ref_price_own, capacity_own, self.E_price_to_volume,
                self.E_price_to_comp_volume
            )

            price_mult_vol_opt = np.sum(price_opt_own_liter * vol_own_opt)
            total_own_volume_opt += vol_own_opt.sum()
            total_price_mult_vol_opt += price_mult_vol_opt

        weighted_avg_opt_price = total_price_mult_vol_opt / (total_own_volume_opt * 100)
        return weighted_avg_opt_price

    def create_constraints(self, tolerance: float = 0.005) -> list:
        """
        Create all nonlinear constraints for the optimization.

        Args:
            tolerance: Tolerance for PINC constraint

        Returns:
            List of NonlinearConstraint objects
        """
        # Industry volume constraint
        ind_vol_nlc_monthly = NonlinearConstraint(
            self.industry_volume_constraint_monthly,
            0.99 * (self.total_ref_industry_volume * (1 - 0.56 * self.target_delta)),
            np.inf
        )

        # Own volume constraint
        own_vol_nlc_monthly = NonlinearConstraint(
            self.own_volume_constraint_monthly,
            0.99 * self.total_ref_own_volume,
            1.05 * self.total_ref_own_volume
        )

        # Market share constraint
        market_share_nlc_monthly = NonlinearConstraint(
            self.market_share_constraint_monthly,
            self.ref_market_share - 0.005,
            np.inf
        )

        # Portfolio PINC constraint
        portfolio_pinc_nlc_monthly = NonlinearConstraint(
            self.portfolio_pinc_constraint_monthly,
            ((1 + self.target_delta) * self.weighted_avg_ref_price) - tolerance,
            ((1 + self.target_delta) * self.weighted_avg_ref_price) + tolerance
        )

        return [
            own_vol_nlc_monthly,
            ind_vol_nlc_monthly,
            market_share_nlc_monthly,
            portfolio_pinc_nlc_monthly
        ]
