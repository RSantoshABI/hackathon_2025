import pandas as pd

# TODO Rohan: PRETTY STRAIGHTFORWARD CLASS, ONLY CHANGES REQUIRED
# MIGHT BE FOR COLUMN NAMES


class DataProcessor:
    """
    Helps process ABI and Competitor Data
    """

    def __init__(
            self,
            market_data: pd.DataFrame,
            elasticity_data: pd.DataFrame
    ):
        self.market_data = market_data
        self.elasticity_data = elasticity_data

        self.abi_data = market_data[market_data['company'] == 'ABI'].copy()
        self.competitor_data = market_data[
            market_data['company'] != 'ABI'].copy()

        self.abi_skus = self.abi_data['sku'].tolist()
        self.all_skus = market_data['sku'].tolist()

        self.abi_indices = {sku: idx for idx, sku in enumerate(self.abi_skus)}
        self.all_sku_indices = {
            sku: idx for idx, sku in enumerate(self.all_skus)}

        self.segment_hierarchy = {
            'Value': 1,
            'Core': 2,
            'Core+': 3,
            'Premium': 4,
            'Super Premium': 5
        }

        self.size_hierarchy = {
            'Small': 3,
            'Regular': 2,
            'Large': 1
        }

        self._build_group_mappings()

    def _build_group_mappings(self):
        """
        Builds mappings for segments and sizes
        """
        self.segment_groups = {}
        for segment in self.segment_hierarchy.keys():
            segment_skus = self.abi_data[
                self.abi_data['segment'] == segment]['sku'].tolist()
            if segment_skus:
                indices = [self.abi_indices[sku] for sku in segment_skus]
                self.segment_groups[segment] = {
                    'indices': indices,
                    'skus': segment_skus,
                    'reference_nr_per_hl': self.abi_data[
                        self.abi_data['segment'] == segment][
                            'nr_per_hl'].values
                }

        self.size_groups = {}
        for size in self.size_hierarchy.keys():
            size_skus = self.abi_data[
                self.abi_data['size_hierarchy'] == size]['sku'].tolist()
            if size_skus:
                indices = [self.abi_indices[sku] for sku in size_skus]
                self.size_groups[size] = {
                    'indices': indices,
                    'skus': size_skus,
                    'reference_nr_per_hl': self.abi_data[
                        self.abi_data[
                            'size_hierarchy'] == size]['nr_per_hl'].values
                }

    def get_abi_reference_data(self):
        """
        Extracts the reference prices/volumes for ABI SKUs
        """
        reference_prices = self.abi_data['price'].values
        reference_volumes = self.abi_data['volume'].values
        reference_nr_per_hl = self.abi_data['nr_per_hl'].values
        return reference_prices, reference_volumes, reference_nr_per_hl

    def get_total_industry_volume(self):
        """
        Calculates and returns the total industry volume
        """
        return self.market_data["volume"].sum()

    def get_abi_market_share(self):
        """
        Calculates ABI market share
        """
        abi_volume = self.abi_data['volume'].sum()
        total_volume = self.get_total_industry_volume()
        return abi_volume/total_volume
