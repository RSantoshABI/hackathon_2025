class HierarchicalPooling:

    def __init__(self, df, min_points):
        self.df = df
        self.min_points = min_points

    
    def create_sku_identifier(self):
        sku_columns = ['brand',	'sub_brand', 'package',	'package_type',	'capacity_number']
        
        self.df['sku'] = (self.df['brand'].astype(str) + ' ' + 
                    self.df['sub_brand'].astype(str) + ' ' + 
                    self.df['package'].astype(str) + ' ' + 
                    self.df['package_type'].astype(str) + ' ' + 
                    self.df['capacity_number'].astype(str))
        return self.df
    
    
    def create_hierarchy_mapping(self):
        sku_counts = self.df.groupby('sku').size()
        
        self.sku_mapping = {}
        self.reverse_mapping = {}
        
        for sku in sku_counts.index:
            if sku_counts[sku] >= self.min_points:
                self.sku_mapping[sku] = sku
                if sku not in self.reverse_mapping:
                    self.reverse_mapping[sku] = []
                self.reverse_mapping[sku].append(sku)
            else:
                sku_row = self.df[self.df['sku'] == sku].iloc[0]
                hierarchy_levels = self._get_hierarchy_levels(sku_row)
                effective_sku = None
                
                for level in hierarchy_levels[1:]:
                    level_data = self.df[self.df.apply(lambda x: level in self._get_hierarchy_levels(x), axis=1)]
                    if len(level_data) >= self.min_points:
                        effective_sku = level
                        break
            
                if effective_sku is None:
                    effective_sku = str(sku_row['brand'])
                self.sku_mapping[sku] = effective_sku
                if effective_sku not in self.reverse_mapping:
                    self.reverse_mapping[effective_sku] = []
                self.reverse_mapping[effective_sku].append(sku)
        
        return self.sku_mapping, self.reverse_mapping

        
    def _get_hierarchy_levels(self,row):
        brand = str(row['brand'])
        subbrand = str(row['sub_brand'])
        package = str(row['package'])
        package_type = str(row['package_type'])
        capacity = str(row['capacity_number'])
        
        levels = [
            f"{brand} {subbrand} {package} {package_type} {capacity}",
            f"{brand} {subbrand} {package} {package_type}",
            f"{brand} {subbrand} {package}",
            f"{brand} {subbrand}",
            f"{brand}",
            ]
            
        return levels