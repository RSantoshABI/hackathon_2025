import tensorflow as tf
import pandas as pd
import logging
from data_processor import DataProcessor
from constraints import (
    BaseConstraint,
    IndustryVolumeConstraint,
    VolumeTargetConstraint,
    MACOConstraint,
    PINCConstraint,
    MarketShareConstraint,
    PriceBoundsConstraint,
    SegmentConstraint,
    SizeConstraint
)
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceOptimizer:

    def __init__(
            self,
            market_data: pd.DataFrame,
            elasticity_data: pd.DataFrame, 
            pinc_value: float,
            pricing_multiple: float = 50.0
    ):
        """
        Initialize ABI-specific optimizer

        Args:
            market_data: Market data including ABI and competitors (must include 'nr_per_hl' column)
            elasticity_data: Cross-price elasticity data
            pinc_value: Portfolio price increase target (0-6%)
            pricing_multiple: Multiple for price rounding (default: 50)
        """
        self.data_processor = DataProcessor(market_data, elasticity_data)
        self.pinc_value = pinc_value
        self.pricing_multiple = pricing_multiple

        reference_prices, reference_volumes, reference_nr_per_hl = self.data_processor.get_abi_reference_data()
        self.reference_prices = tf.constant(reference_prices, dtype=tf.float32)
        self.reference_volumes = tf.constant(reference_volumes, dtype=tf.float32)
        self.reference_nr_per_hl = tf.constant(reference_nr_per_hl, dtype=tf.float32)
        self.sku_names = self.data_processor.abi_skus

        self.elasticity_matrix = self._build_elasticity_matrix()

        self.hard_constraints: List[BaseConstraint] = []
        self.soft_constraints: List[BaseConstraint] = []

        self.prices = None
        self.optimizer = None

    def _build_elasticity_matrix(self) -> np.ndarray:
        """Build elasticity matrix from elasticity data"""
        n_abi = len(self.data_processor.abi_skus)
        n_total = len(self.data_processor.all_skus)

        E_full = np.zeros((n_total, n_total))

        for _, row in self.data_processor.elasticity_data.iterrows():
            if (row['target_sku'] in self.data_processor.all_sku_indices and
                row['other_sku'] in self.data_processor.all_sku_indices):

                i = self.data_processor.all_sku_indices[row['target_sku']]
                j = self.data_processor.all_sku_indices[row['other_sku']]
                E_full[i, j] = row['elasticity']

        return E_full

    def _round_to_multiple(self, prices: np.ndarray, multiple: float = None) -> np.ndarray:
        if multiple is None:
            multiple = self.pricing_multiple

        price_changes = prices - self.reference_prices.numpy()
        rounded_changes = np.round(price_changes / multiple) * multiple
        return self.reference_prices.numpy() + rounded_changes

    def setup_default_constraints(self):

        self.hard_constraints = [
            IndustryVolumeConstraint(self.data_processor, self.elasticity_matrix),
            MACOConstraint(self.data_processor),
            VolumeTargetConstraint(self.data_processor),
            PINCConstraint(self.data_processor, self.pinc_value),
            MarketShareConstraint(self.data_processor, self.elasticity_matrix),
            PriceBoundsConstraint(
                lower_bounds=self.reference_prices.numpy() - 300,
                upper_bounds=self.reference_prices.numpy() + 500
            )
        ]

        self.soft_constraints = [
            SegmentConstraint(self.data_processor, penalty_weight=1000.0),
            SizeConstraint(self.data_processor, penalty_weight=1000.0)
        ]

    def optimize(
            self,
            learning_rate: float = 0.01,
            max_iterations: int = 2000, 
            tolerance: float = 1e-6,
            verbose: bool = True, 
            apply_rounding: bool = True
    ) -> Dict[str, Any]:

        if not self.hard_constraints and not self.soft_constraints:
            self.setup_default_constraints()

        self.prices = tf.Variable(self.reference_prices, dtype=tf.float32, name='prices')
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        prev_loss = tf.constant(float('inf'))
        results = {
            'success': False,
            'iterations': 0,
            'final_loss': None,
            'optimal_prices': None,
            'hard_constraint_violations': {},
            'soft_constraint_violations': {},
            'convergence_history': []
        }

        for iteration in range(max_iterations):
            with tf.GradientTape() as tape:
                loss = self._compute_objective()

            gradients = tape.gradient(loss, [self.prices])
            self.optimizer.apply_gradients(zip(gradients, [self.prices]))

            # Check convergence
            loss_change = tf.abs(prev_loss - loss)
            results['convergence_history'].append(float(loss))

            if verbose and iteration % 200 == 0:
                print(f"Iteration {iteration}: Loss = {float(loss):.6f}")

            if loss_change < tolerance:
                results['success'] = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_loss = loss

        raw_optimal_prices = self.prices.numpy()

        if apply_rounding:
            rounded_prices = self._round_to_multiple(raw_optimal_prices)
            results['optimal_prices'] = rounded_prices
            results['raw_optimal_prices'] = raw_optimal_prices

            self._validate_rounded_prices(rounded_prices, results)
        else:
            results['optimal_prices'] = raw_optimal_prices

        results['iterations'] = iteration + 1
        results['final_loss'] = float(loss)

        self._check_constraint_violations(results)
        return results

    def _compute_objective(self) -> tf.Tensor:
        # TODO ROHAN MODIFY THIS FUNCTION - USE MACO INSTEAD OF REVENUE
        demand = self._compute_abi_demand()
        revenue = tf.reduce_sum(self.prices * demand)
        objective = -revenue

        constraint_kwargs = {
            'demand': demand,
            'reference_volumes': self.reference_volumes
        }

        for constraint in self.hard_constraints:
            penalty = constraint.compute_penalty(self.prices, **constraint_kwargs)
            objective += penalty

        for constraint in self.soft_constraints:
            penalty = constraint.compute_penalty(self.prices, **constraint_kwargs)
            objective += penalty

        return objective

    def _compute_abi_demand(self) -> tf.Tensor:
        price_ratios = self.prices / self.reference_prices
        log_ratios = tf.math.log(price_ratios)

        n_abi = len(self.data_processor.abi_skus)
        E_abi = self.elasticity_matrix[:n_abi, :n_abi]

        demand = self.reference_volumes * tf.exp(
            tf.linalg.matvec(tf.constant(E_abi, dtype=tf.float32), log_ratios)
        )

        return demand

    def _compute_demand_from_prices(self, prices: tf.Tensor) -> tf.Tensor:
        price_ratios = prices / self.reference_prices
        log_ratios = tf.math.log(price_ratios)

        n_abi = len(self.data_processor.abi_skus)
        E_abi = self.elasticity_matrix[:n_abi, :n_abi]

        demand = self.reference_volumes * tf.exp(
            tf.linalg.matvec(tf.constant(E_abi, dtype=tf.float32), log_ratios)
        )

        return demand

    def _validate_rounded_prices(self, rounded_prices: np.ndarray, results: Dict[str, Any]):        
        # Update TensorFlow variable with rounded prices for constraint checking
        rounded_tensor = tf.constant(rounded_prices, dtype=tf.float32)
        constraint_kwargs = {
            'demand': self._compute_demand_from_prices(rounded_tensor),
            'reference_volumes': self.reference_volumes
        }

        violations = {}
        for constraint in self.hard_constraints:
            is_satisfied = constraint.is_satisfied(rounded_tensor, **constraint_kwargs)
            violations[constraint.name] = not bool(is_satisfied)

        results['rounding_validation'] = {
            'violations_after_rounding': violations,
            'rounding_applied': True,
            'pricing_multiple': self.pricing_multiple
        }

        critical_violations = [name for name, violated in violations.items() if violated]
        if critical_violations:
            logger.warning(f"Rounding violated constraints: {critical_violations}")

    def _check_constraint_violations(self, results: Dict[str, Any]):
        final_prices = tf.constant(results['optimal_prices'], dtype=tf.float32)
        constraint_kwargs = {
            'demand': self._compute_demand_from_prices(final_prices),
            'reference_volumes': self.reference_volumes
        }

        for constraint in self.hard_constraints:
            is_satisfied = constraint.is_satisfied(final_prices, **constraint_kwargs)
            results['hard_constraint_violations'][constraint.name] = not bool(is_satisfied)

        for constraint in self.soft_constraints:
            is_satisfied = constraint.is_satisfied(final_prices, **constraint_kwargs)  
            results['soft_constraint_violations'][constraint.name] = not bool(is_satisfied)

    def get_optimization_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        if results['optimal_prices'] is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        optimal_prices = results['optimal_prices']
        optimal_demand = self._compute_demand_from_prices(tf.constant(optimal_prices)).numpy()

        # TODO ROHAN CHANGE THIS ACCORDING TO NR/HL SCALING LOGIC
        # Calculate new NR/HL (assuming proportional scaling with price changes)
        price_ratios = optimal_prices / self.reference_prices.numpy()
        new_nr_per_hl = self.reference_nr_per_hl.numpy() * price_ratios

        summary_df = pd.DataFrame({
            'sku': self.sku_names,
            'segment': self.data_processor.abi_data['segment'].values,
            'size_hierarchy': self.data_processor.abi_data['size_hierarchy'].values,
            'reference_price': self.reference_prices.numpy(),
            'reference_volume': self.reference_volumes.numpy(),
            'reference_nr_per_hl': self.reference_nr_per_hl.numpy(),
            'optimized_price': optimal_prices,
            'optimized_volume': optimal_demand,
            'optimized_nr_per_hl': new_nr_per_hl,
            'price_change': optimal_prices - self.reference_prices.numpy(),
            'price_change_pct': ((optimal_prices - self.reference_prices.numpy()) / 
                               self.reference_prices.numpy()) * 100,
            'volume_change_pct': ((optimal_demand - self.reference_volumes.numpy()) / 
                                self.reference_volumes.numpy()) * 100,
            'nr_per_hl_change_pct': ((new_nr_per_hl - self.reference_nr_per_hl.numpy()) /
                                   self.reference_nr_per_hl.numpy()) * 100,
            'revenue_change_pct': (((optimal_prices * optimal_demand) - 
                                  (self.reference_prices.numpy() * self.reference_volumes.numpy())) /
                                 (self.reference_prices.numpy() * self.reference_volumes.numpy())) * 100
        })

        if 'raw_optimal_prices' in results:
            raw_prices = results['raw_optimal_prices']
            summary_df['raw_optimized_price'] = raw_prices
            summary_df['rounding_adjustment'] = optimal_prices - raw_prices

            # Check if price changes are multiples of the specified value
            price_changes = optimal_prices - self.reference_prices.numpy()
            summary_df['multiple_compliance'] = np.abs(price_changes % self.pricing_multiple) < 0.01

        return summary_df