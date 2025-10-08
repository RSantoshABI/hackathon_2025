import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import logging
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseConstraint(ABC):

    @abstractmethod
    def compute_penalty(self, prices: tf.Tensor, **kwargs):
        pass

    @abstractmethod
    def is_satisfied(self, prices: tf.Tensor, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class PriceBoundsConstraint(BaseConstraint):
    """
    Price should not increase more than 500, decrease more than 300.
    HARD CONSTRAINT
    """

    def __init__(
            self,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray,
            penalty_weights: float = 10000.0
    ):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.penalty_weights = penalty_weights

    def compute_penalty(self, prices: tf.Tensor, **kwargs):
        lower_violation = tf.maximum(0.0, self.lower_bounds - prices)
        upper_violation = tf.maximum(0.0, prices - self.upper_bounds)
        penalty = tf.reduce_sum(
            tf.square(lower_violation) + tf.square(upper_violation)
            )
        return self.penalty_weights * penalty

    def is_satisfied(self, prices, **kwargs):
        within_bounds = tf.logical_and(
            tf.greater_equal(prices, self.lower_bounds),
            tf.less_equal(prices, self.upper_bounds)
        )
        return tf.reduce_all(within_bounds)

    @property
    def name(self) -> str:
        return "PriceBounds"


class IndustryVolumeConstraint(BaseConstraint):
    """
    Industry Volume should not decline by more than 1%
    HARD CONSTRAINT
    """

    def __init__(
            self,
            data_processor: DataProcessor,
            elasticity_matrix: np.ndarray,
            max_decline_pct: float = 0.01,
            penalty_weight: float = 10000.0
    ):
        self.data_processor = data_processor
        self.elasticity_matrix = tf.constant(elasticity_matrix, dtype=tf.float32)
        self.max_decline_pct = max_decline_pct
        self.penalty_weight = penalty_weight

        self.current_industry_volume = tf.constant(
            data_processor.get_total_industry_volume(), dtype=tf.float32
        )

        self.min_allowed_volume = self.current_industry_volume * (1 - max_decline_pct)

        all_volumes = data_processor.market_data['volume'].values
        all_prices = data_processor.market_data['price'].values
        self.all_reference_volumes = tf.constant(all_volumes, dtype=tf.float32)
        self.all_reference_prices = tf.constant(all_prices, dtype=tf.float32)

    def _compute_new_industry_volume(
            self,
            abi_prices: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculates the industry volume *AFTER* ABI price changes
        """
        full_prices = tf.concat([abi_prices, self.all_reference_prices[10:]], axis=0)
        price_ratios = full_prices / self.all_reference_prices
        log_ratios = tf.math.log(price_ratios)
        new_volumes = self.all_reference_volumes * tf.exp(
            tf.linalg.matvec(self.elasticity_matrix, log_ratios)
        )
        return tf.reduce_sum(new_volumes)

    def compute_penalty(
            self,
            prices: tf.Tensor,
            **kwargs
    ) -> tf.Tensor:
        new_industry_volume = self._compute_new_industry_volume(prices)
        violation = tf.maximum(0.0, self.min_allowed_volume - new_industry_volume)
        return self.penalty_weight * tf.square(violation)

    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        new_industry_volume = self._compute_new_industry_volume(prices)
        return tf.greater_equal(new_industry_volume, self.min_allowed_volume)

    @property
    def name(self) -> str:
        return "IndustryVolume"


class VolumeTargetConstraint(BaseConstraint):
    """
    Volume should not decrease >1% or increase >5%
    HARD CONSTRAINT
    """

    def __init__(
            self,
            data_processor: DataProcessor,
            max_decrease_pct: float = 0.01,
            max_increase_pct: float = 0.05,
            penalty_weight: float = 10000.0
    ):
        self.data_processor = data_processor
        self.max_decrease_pct = max_decrease_pct
        self.max_increase_pct = max_increase_pct
        self.penalty_weight = penalty_weight

        _, reference_volumes, _ = data_processor.get_abi_reference_data()
        self.reference_volumes = tf.constant(reference_volumes, dtype=tf.float32)
        self.total_base_volume = tf.reduce_sum(self.reference_volumes)

        self.min_allowed_volume = self.total_base_volume * (1 - max_decrease_pct)
        self.max_allowed_volume = self.total_base_volume * (1 + max_increase_pct)

    def compute_penalty(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        demand = kwargs.get('demand', self.reference_volumes)
        total_new_volume = tf.reduce_sum(demand)

        lower_violation = tf.maximum(0.0, self.min_allowed_volume - total_new_volume)
        upper_violation = tf.maximum(0.0, total_new_volume - self.max_allowed_volume)

        return self.penalty_weight * (tf.square(lower_violation) + tf.square(upper_violation))

    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        demand = kwargs.get('demand', self.reference_volumes)
        total_new_volume = tf.reduce_sum(demand)

        return tf.logical_and(
            tf.greater_equal(total_new_volume, self.min_allowed_volume),
            tf.less_equal(total_new_volume, self.max_allowed_volume)
        )

    @property
    def name(self) -> str:
        return "VolumeTarget"


class MACOConstraint(BaseConstraint):
    """
    Target MACO must be higher than base MACO
    HARD CONSTRAINT
    """
    # TODO ROHAN THIS IS WRITTEN FOR REVENUE NEED TO MODIFY FOR MACO
    
    def __init__(
            self,
            data_processor: DataProcessor,
            penalty_weight: float = 10000.0
    ):
        self.data_processor = data_processor
        self.penalty_weight = penalty_weight

        reference_prices, reference_volumes, _ = data_processor.get_abi_reference_data()
        self.reference_prices = tf.constant(reference_prices, dtype=tf.float32)
        self.reference_volumes = tf.constant(reference_volumes, dtype=tf.float32)
        self.base_revenue = tf.reduce_sum(self.reference_prices * self.reference_volumes)
    
    def compute_penalty(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        demand = kwargs.get('demand', self.reference_volumes)
        new_revenue = tf.reduce_sum(prices * demand)
        violation = tf.maximum(0.0, self.base_revenue - new_revenue)
        return self.penalty_weight * tf.square(violation)
    
    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        demand = kwargs.get('demand', self.reference_volumes)
        new_revenue = tf.reduce_sum(prices * demand)
        return tf.greater_equal(new_revenue, self.base_revenue)
    
    @property
    def name(self) -> str:
        return "MACOTarget"


class PINCConstraint(BaseConstraint):
    """
    Portfolio price increase should be between 0-6%
    HARD CONSTRAINT
    """
    def __init__(
            self,
            data_processor: DataProcessor,
            pinc_value: float,
            max_pinc: float = 0.06,
            penalty_weight: float = 10000.0
    ):
        self.data_processor = data_processor
        self.pinc_value = pinc_value
        self.max_pinc = max_pinc
        self.penalty_weight = penalty_weight

        reference_prices, reference_volumes, _ = data_processor.get_abi_reference_data()
        self.reference_prices = tf.constant(reference_prices, dtype=tf.float32)
        self.reference_volumes = tf.constant(reference_volumes, dtype=tf.float32)

        self.base_revenue = tf.reduce_sum(self.reference_prices * self.reference_volumes)
        self.target_revenue = self.base_revenue * (1 + pinc_value)

    def compute_penalty(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        new_revenue = tf.reduce_sum(prices * self.reference_volumes)
        current_pinc = (new_revenue - self.base_revenue) / self.base_revenue
        lower_violation = tf.maximum(0.0, -current_pinc)
        upper_violation = tf.maximum(0.0, current_pinc - self.max_pinc)
        return self.penalty_weight * (tf.square(lower_violation) + tf.square(upper_violation))

    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        new_revenue = tf.reduce_sum(prices * self.reference_volumes)
        current_pinc = (new_revenue - self.base_revenue) / self.base_revenue
        return tf.logical_and(
            tf.greater_equal(current_pinc, 0.0),
            tf.less_equal(current_pinc, self.max_pinc)
        )

    @property
    def name(self) -> str:
        return "PortfolioPINC"


class MarketShareConstraint(BaseConstraint):
    """
    ABI market share should not drop by more than 0.5%
    HARD CONSTRAINT
    """
    def __init__(
            self,
            data_processor: DataProcessor,
            elasticity_matrix: np.ndarray,
            max_drop_pct: float = 0.005,
            penalty_weight: float = 10000.0
    ):
        self.data_processor = data_processor
        self.elasticity_matrix = tf.constant(elasticity_matrix, dtype=tf.float32)
        self.max_drop_pct = max_drop_pct
        self.penalty_weight = penalty_weight

        self.current_market_share = data_processor.get_abi_market_share()
        self.min_market_share = self.current_market_share - max_drop_pct

        self.abi_reference_volumes = tf.constant(
            data_processor.abi_data['volume'].values, dtype=tf.float32
        )
        self.total_reference_volumes = tf.constant(
            data_processor.market_data['volume'].values, dtype=tf.float32
        )

    def compute_new_market_share(self, abi_prices: tf.Tensor) -> tf.Tensor:
        # This would need the full elasticity calculation
        # For now, simplified version assuming only ABI volumes change
        demand = self.abi_reference_volumes  # This should come from elasticity calculation
        new_abi_volume = tf.reduce_sum(demand)
        competitor_volume = tf.reduce_sum(self.total_reference_volumes) - tf.reduce_sum(self.abi_reference_volumes)
        total_new_volume = new_abi_volume + competitor_volume
        return new_abi_volume / total_new_volume
    
    def compute_penalty(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        new_market_share = self.compute_new_market_share(prices)
        violation = tf.maximum(0.0, self.min_market_share - new_market_share)
        return self.penalty_weight * tf.square(violation)
    
    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        new_market_share = self.compute_new_market_share(prices)
        return tf.greater_equal(new_market_share, self.min_market_share)
    
    @property
    def name(self) -> str:
        return "MarketShare"


class SegmentConstraint(BaseConstraint):
    """
    NR/HL hierarchy: Value < Core < Core+ < Premium < Super Premium
    SOFT CONSTRAINT
    """
    # TODO ROHAN THE COMPUTE NEW NR PER HL FOR SEGMENT FUNCTION IS WRONG,
    # IT NEEDS TO BE REDONE WITH THE LOGIC WITH WHICH WE WANT THE NR/HL TO SCALE

    def __init__(
            self,
            data_processor: DataProcessor,
            penalty_weight: float = 1000.0, 
            use_volume_weighted: bool = True
    ):
        self.data_processor = data_processor
        self.penalty_weight = penalty_weight
        self.use_volume_weighted = use_volume_weighted

        reference_prices, reference_volumes, reference_nr_per_hl = data_processor.get_abi_reference_data()
        self.reference_prices = tf.constant(reference_prices, dtype=tf.float32)
        self.reference_volumes = tf.constant(reference_volumes, dtype=tf.float32)
        self.reference_nr_per_hl = tf.constant(reference_nr_per_hl, dtype=tf.float32)

        self.segment_groups = {}
        for segment, group_data in data_processor.segment_groups.items():
            self.segment_groups[segment] = {
                'indices': tf.constant(group_data['indices'], dtype=tf.int32),
                'reference_nr_per_hl': tf.constant(group_data['reference_nr_per_hl'], dtype=tf.float32)
            }

    def _compute_new_nr_per_hl_for_segment(self, prices: tf.Tensor, segment: str) -> tf.Tensor:
        """
        Compute new NR/HL for a segment based on price changes

        This assumes NR/HL scales proportionally with price changes
        More sophisticated modeling could be implemented here
        """
        if segment not in self.segment_groups:
            return tf.constant(0.0, dtype=tf.float32)

        group_data = self.segment_groups[segment]
        indices = group_data['indices']
        reference_nr_per_hl = group_data['reference_nr_per_hl']

        segment_prices = tf.gather(prices, indices)
        segment_reference_prices = tf.gather(self.reference_prices, indices)

        price_ratios = segment_prices / segment_reference_prices

        # Assume NR/HL changes proportionally with price changes
        # In practice, you might have a more sophisticated model
        new_nr_per_hl = reference_nr_per_hl * price_ratios

        if self.use_volume_weighted:
            # Volume-weighted average NR/HL for the segment
            segment_volumes = tf.gather(self.reference_volumes, indices)
            total_volume = tf.reduce_sum(segment_volumes)

            if total_volume > 0:
                weighted_nr_per_hl = tf.reduce_sum(new_nr_per_hl * segment_volumes) / total_volume
            else:
                weighted_nr_per_hl = tf.reduce_mean(new_nr_per_hl)
        else:
            # Simple average
            weighted_nr_per_hl = tf.reduce_mean(new_nr_per_hl)

        return weighted_nr_per_hl

    def compute_penalty(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        penalty = 0.0

        segment_nr_hl = {}
        for segment in self.data_processor.segment_hierarchy.keys():
            if segment in self.segment_groups:
                segment_nr_hl[segment] = self._compute_new_nr_per_hl_for_segment(prices, segment)

        hierarchy_order = ['Value', 'Core', 'Core+', 'Premium', 'Super Premium']
        for i in range(len(hierarchy_order) - 1):
            current_segment = hierarchy_order[i]
            next_segment = hierarchy_order[i + 1]

            if current_segment in segment_nr_hl and next_segment in segment_nr_hl:
                violation = tf.maximum(0.0, 
                    segment_nr_hl[current_segment] - segment_nr_hl[next_segment] + 0.1)
                penalty += tf.square(violation)
        return self.penalty_weight * penalty

    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        """Check if segment hierarchy is maintained"""
        segment_nr_hl = {}
        for segment in self.data_processor.segment_hierarchy.keys():
            if segment in self.segment_groups:
                segment_nr_hl[segment] = self._compute_new_nr_per_hl_for_segment(prices, segment)

        hierarchy_order = ['Value', 'Core', 'Core+', 'Premium', 'Super Premium']
        all_satisfied = tf.constant(True, dtype=tf.bool)

        for i in range(len(hierarchy_order) - 1):
            current_segment = hierarchy_order[i]
            next_segment = hierarchy_order[i + 1]

            if current_segment in segment_nr_hl and next_segment in segment_nr_hl:
                is_ordered = tf.less(segment_nr_hl[current_segment] + 0.01, 
                                   segment_nr_hl[next_segment])
                all_satisfied = tf.logical_and(all_satisfied, is_ordered)
        return all_satisfied

    @property
    def name(self) -> str:
        return "SegmentHierarchy"


class SizeConstraint(BaseConstraint):
    """
    NR/HL size hierarchy: Small > Regular > Large
    SOFT CONSTRAINTS
    """
    def __init__(
            self,
            data_processor: DataProcessor,
            penalty_weight: float = 1000.0,
            use_volume_weighted: bool = True
    ):
        self.data_processor = data_processor
        self.penalty_weight = penalty_weight
        self.use_volume_weighted = use_volume_weighted

        reference_prices, reference_volumes, reference_nr_per_hl = data_processor.get_abi_reference_data()
        self.reference_prices = tf.constant(reference_prices, dtype=tf.float32)
        self.reference_volumes = tf.constant(reference_volumes, dtype=tf.float32)
        self.reference_nr_per_hl = tf.constant(reference_nr_per_hl, dtype=tf.float32)

        self.size_groups = {}
        for size, group_data in data_processor.size_groups.items():
            self.size_groups[size] = {
                'indices': tf.constant(group_data['indices'], dtype=tf.int32),
                'reference_nr_per_hl': tf.constant(group_data['reference_nr_per_hl'], dtype=tf.float32)
            }

    def _compute_new_nr_per_hl_for_size(self, prices: tf.Tensor, size: str) -> tf.Tensor:
        """Compute new NR/HL for a size group based on price changes"""
        if size not in self.size_groups:
            return tf.constant(0.0, dtype=tf.float32)

        group_data = self.size_groups[size]
        indices = group_data['indices']
        reference_nr_per_hl = group_data['reference_nr_per_hl']

        # Get current and reference prices for this size
        size_prices = tf.gather(prices, indices)
        size_reference_prices = tf.gather(self.reference_prices, indices)

        # Calculate price change ratios
        price_ratios = size_prices / size_reference_prices

        # Assume NR/HL changes proportionally with price changes
        new_nr_per_hl = reference_nr_per_hl * price_ratios

        if self.use_volume_weighted:
            # Volume-weighted average NR/HL for the size group
            size_volumes = tf.gather(self.reference_volumes, indices)
            total_volume = tf.reduce_sum(size_volumes)

            if total_volume > 0:
                weighted_nr_per_hl = tf.reduce_sum(new_nr_per_hl * size_volumes) / total_volume
            else:
                weighted_nr_per_hl = tf.reduce_mean(new_nr_per_hl)
        else:
            # Simple average
            weighted_nr_per_hl = tf.reduce_mean(new_nr_per_hl)

        return weighted_nr_per_hl

    def compute_penalty(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        """Compute penalty for size hierarchy violations"""
        penalty = 0.0

        # Calculate NR/HL for each size
        size_nr_hl = {}
        for size in self.data_processor.size_hierarchy.keys():
            if size in self.size_groups:
                size_nr_hl[size] = self._compute_new_nr_per_hl_for_size(prices, size)

        # Check hierarchy: Small > Regular > Large
        hierarchy_order = ['Small', 'Regular', 'Large']

        for i in range(len(hierarchy_order) - 1):
            current_size = hierarchy_order[i]
            next_size = hierarchy_order[i + 1]

            if current_size in size_nr_hl and next_size in size_nr_hl:
                # Penalty if hierarchy is violated (current <= next when it should be >)
                # Add small buffer (0.1) to allow for minor numerical differences
                violation = tf.maximum(0.0,
                    size_nr_hl[next_size] - size_nr_hl[current_size] + 0.1)
                penalty += tf.square(violation)

        return self.penalty_weight * penalty

    def is_satisfied(self, prices: tf.Tensor, **kwargs) -> tf.Tensor:
        """Check if size hierarchy is maintained"""
        # Calculate NR/HL for each size
        size_nr_hl = {}
        for size in self.data_processor.size_hierarchy.keys():
            if size in self.size_groups:
                size_nr_hl[size] = self._compute_new_nr_per_hl_for_size(prices, size)

        # Check all adjacent pairs in hierarchy
        hierarchy_order = ['Small', 'Regular', 'Large']  
        all_satisfied = tf.constant(True, dtype=tf.bool)

        for i in range(len(hierarchy_order) - 1):
            current_size = hierarchy_order[i]
            next_size = hierarchy_order[i + 1]

            if current_size in size_nr_hl and next_size in size_nr_hl:
                # Check if current > next (with small tolerance)
                is_ordered = tf.greater(size_nr_hl[current_size], 
                                      size_nr_hl[next_size] + 0.01)
                all_satisfied = tf.logical_and(all_satisfied, is_ordered)

        return all_satisfied

    @property
    def name(self) -> str:
        return "SizeHierarchy"


###########################
# IGNORE EVERYTHING BELOW #
###########################


# class BaseConstraint(ABC):

#     @abstractmethod
#     def compute_penalty(self, prices: tf.Tensor, **kwargs):
#         pass

#     @abstractmethod
#     def is_satisfied(self, prices: tf.Tensor, **kwargs):
#         pass

#     @property
#     @abstractmethod
#     def name(self) -> str:
#         pass


# class PriceBoundsConstraint(BaseConstraint):
#     """
#     Price should not increase more than 500, decrease more than 300.
#     """

#     def __init__(
#             self,
#             lower_bounds: np.ndarray,
#             upper_bounds: np.ndarray,
#             penalty_weights: float = 1000.0
#     ):
#         self.lower_bounds = lower_bounds
#         self.upper_bounds = upper_bounds
#         self.penalty_weights = penalty_weights

#     def compute_penalty(self, prices: tf.Tensor, **kwargs):
#         lower_violation = tf.maximum(0.0, self.lower_bounds - prices)
#         upper_violation = tf.maximum(0.0, prices - self.upper_bounds)
#         penalty = tf.reduce_sum(
#             tf.square(lower_violation) + tf.square(upper_violation)
#             )
#         return self.penalty_weights + penalty

#     def is_satisfied(self, prices, **kwargs):
#         within_bounds = tf.logical_and(
#             tf.greater_equal(prices, self.lower_bounds),
#             tf.less_equal(prices, self.upper_bounds)
#         )
#         return tf.reduce_all(within_bounds)

#     @property
#     def name(self) -> str:
#         return "PriceBounds"


# class IndustryVolumeConstraint(BaseConstraint):
#     """
#     Industry volume should not increase more than 5%, decrease more than 1%
#     """

#     def __init__(
#             self,
#             reference_prices: np.ndarray,
#             max_increase_pct: float = 0.05,
#             max_decrease_pct: float = 0.01,
#     ):
#         self.reference_prices = reference_prices
#         self.max_increase_pct = max_increase_pct
#         self.max_decrease_pct = max_decrease_pct

#     def compute_penalty(self, prices, **kwargs):
#         pass

