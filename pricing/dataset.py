from hierarchical_pooling import HierarchicalPooling
import pandas as pd
import numpy as np

class Dataset:

    def __init__(self, sellout_df, ihs_df, min_points=18):
        self.sellout_df = sellout_df
        self.ihs_df = ihs_df
        self.min_points = min_points
        
        hierarchical_pooling = HierarchicalPooling(self.sellout_df, self.min_points)
        self.updated_df = hierarchical_pooling.create_sku_identifier()
        self.sku_mapping, self.reverse_mapping = hierarchical_pooling.create_hierarchy_mapping()


    def process_data(self):
        processed_sellout_df = self._process_sellout_data()
        processed_ihs_df = self._process_ihs_data()
        
        processed_sellout_df['original_sku'] = processed_sellout_df['sku'] 
        processed_sellout_df['effective_sku'] = processed_sellout_df['sku'].map(self.sku_mapping)
        
        agg_df = processed_sellout_df.groupby(['date', 'effective_sku']).agg({
            'sales_hectoliters': 'sum',
            'avg_price_per_liter': 'mean',
            'weighted_distribution_tdp_reach': 'mean',
            'original_sku': 'first'
            }).reset_index()
        
        self.processed_df = pd.merge(
            agg_df,
            processed_ihs_df,
            on='date',
            how='left',
            indicator=True)
        
        self.processed_df['log_price'] = np.log(self.processed_df['avg_price_per_liter'])
        self.processed_df['log_volume'] = np.log(self.processed_df['sales_hectoliters'])
        self.processed_df['log_distribution'] = np.log(self.processed_df['weighted_distribution_tdp_reach'].clip(lower=0.01))

        self.price_matrix, self.volume_matrix = self._create_price_volume_matrices()
        self.feature_matrix, self.feature_names = self._create_feature_matrix()

        results_dict = self.get_model_inputs()     
        return results_dict


    def _process_sellout_data(self):
        
        sellout_df = self.updated_df.copy()
        sellout_df['date'] = pd.to_datetime(sellout_df['date'])
        sellout_df['date'] = sellout_df['date'].apply(lambda x: x.replace(day=1))
        sellout_df = sellout_df.dropna(subset=['sales_hectoliters'])
        sellout_df = sellout_df.sort_values(['sku', 'date'])
        sellout_df['avg_price_per_liter'] = sellout_df.groupby('sku')['avg_price_per_liter'].fillna(method='ffill')
        sellout_df['weighted_distribution_tdp_reach'] = sellout_df.groupby('sku')['weighted_distribution_tdp_reach'].fillna(method='ffill')
        return sellout_df


    def _process_ihs_data(self):
        
        ihs_df = self.ihs_df.copy()
        ihs_df['date'] = pd.to_datetime(ihs_df['date'])
        as_is_cols = ['inflation_cpi', 'interest_rate', 'unemployment_rate']
        base_adj = ['cpi', 'wholesale_price_index']
        log_transform = ['retail_sales', 'consumption_total', 'domestic_demand', 'gdp_per_capita', 'gdp_real', 'fixed_investment','gross_capital_formation']
        # normalized = ['primary_income']

        macro_cols = [col for col in ihs_df.columns if col != 'date']
        for col in macro_cols:
            if col in as_is_cols:
                ihs_df[col] = ihs_df[col]
            elif col in base_adj:
                ihs_df[col] = ihs_df[col] / ihs_df[col].iloc[0]
            elif col in log_transform:
                ihs_df[col] = np.log(ihs_df[col].clip(lower=0.01))
            else:
                ihs_df.drop(columns=[col], inplace=True)

        return ihs_df


    def _create_feature_matrix(self):
        feature_cols = ['log_distribution']
        unique_dates = self.price_matrix.index
        macro_feature_cols = [col for col in self.processed_df.columns if col not in ['log_price','log_volume','log_distribution', 'date', 'effective_sku', 'sales_hectoliters', 'avg_price_per_liter', 'weighted_distribution_tdp_reach', 'original_sku']]
        feature_cols.extend(macro_feature_cols)
        # feature_matrix = self.processed_df[feature_cols]
        features_by_date = []
        
        for date in unique_dates:
            date_data = self.processed_df[self.processed_df['date'] == date]

            # 1. SKU-specific features (averaged across SKUs for that date)
            avg_distribution = date_data['log_distribution'].mean()
            
            # 2. Macro variables (same for all SKUs on that date)
            macro_values = date_data[macro_feature_cols].iloc[0] if len(date_data) > 0 else pd.Series(
                [0] * len(macro_feature_cols), index=macro_feature_cols)
            
            # Combine into single row
            row_features = [avg_distribution] + macro_values.tolist()
            features_by_date.append(row_features)
        
        # Convert to DataFrame
        feature_matrix = pd.DataFrame(features_by_date, 
                                    index=unique_dates, 
                                    columns=feature_cols)
        
        feature_matrix = feature_matrix.fillna(method='ffill').fillna(method='bfill')
        return feature_matrix, feature_cols
    
    def _create_price_volume_matrices(self):
        price_matrix = self.processed_df.pivot(
            index='date', columns='effective_sku', values='log_price'
        ).fillna(method='ffill').fillna(method='bfill')
        
        volume_matrix = self.processed_df.pivot(
            index='date', columns='effective_sku', values='log_volume'
        ).fillna(method='ffill').fillna(method='bfill')
        return price_matrix, volume_matrix


    def get_model_inputs(self):
        """Return properly formatted inputs for TensorFlow model"""
        return {
            'log_prices': self.price_matrix.values,
            'features': self.feature_matrix.values,
            'log_volumes': self.volume_matrix.values,
            'feature_names': self.feature_names,
            'effective_skus': list(self.price_matrix.columns),
            'n_features': len(self.feature_names)
        }