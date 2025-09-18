import numpy as np
import pandas as pd

def expand_elasticity_matrix_to_original_skus(learned_elasticity_matrix, data_obj):
    """
    Expand elasticity matrix from effective SKUs back to original SKUs and return as DataFrame
    
    Args:
        learned_elasticity_matrix: [n_effective_skus, n_effective_skus] numpy array
        data_obj: Dataset object containing sku_mapping and reverse_mapping
        
    Returns:
        DataFrame with columns: target_sku, other_sku, elasticity
    """
    
    # Get mappings from data object
    sku_mapping = data_obj.sku_mapping  # original_sku -> effective_sku
    reverse_mapping = data_obj.reverse_mapping  # effective_sku -> [list of original_skus]
    
    # Get list of effective SKUs (columns of learned matrix)
    effective_skus = list(set(sku_mapping.values()))
    
    # Get all original SKUs
    all_original_skus = list(sku_mapping.keys())
    
    # Create mapping from effective SKU to its index in the learned matrix
    effective_sku_to_index = {sku: idx for idx, sku in enumerate(effective_skus)}
    
    # Create results list
    results = []
    
    # For each pair of original SKUs
    for target_sku in all_original_skus:
        for other_sku in all_original_skus:
            
            # Find which effective SKUs these original SKUs map to
            target_effective_sku = sku_mapping[target_sku]
            other_effective_sku = sku_mapping[other_sku]
            
            # Get indices in the learned matrix
            target_idx = effective_sku_to_index.get(target_effective_sku)
            other_idx = effective_sku_to_index.get(other_effective_sku)
            
            # Get elasticity value from learned matrix
            if target_idx is not None and other_idx is not None:
                elasticity = learned_elasticity_matrix[target_idx, other_idx]
            else:
                elasticity = 0.0  # Default if mapping not found
            
            # Determine if it's own-price or cross-price elasticity
            elasticity_type = "own_price" if target_sku == other_sku else "cross_price"
            
            # Add to results
            results.append({
                'target_sku': target_sku,
                'other_sku': other_sku, 
                'elasticity': elasticity,
                'elasticity_type': elasticity_type  # Optional column for clarity
            })
    
    # Convert to DataFrame
    elasticity_df = pd.DataFrame(results)
    
    # Sort for better readability
    elasticity_df = elasticity_df.sort_values(['target_sku', 'other_sku']).reset_index(drop=True)
    
    return elasticity_df

def old(learned_elasticity_matrix, data_obj):
    
    effective_skus = data_obj.sku_mapping
    reverse_mapping = data_obj.reverse_mapping

    # Get all original SKUs
    all_original_skus = []
    for effective_sku in effective_skus:
        if effective_sku in reverse_mapping:
            all_original_skus.extend(reverse_mapping[effective_sku])
        else:
            all_original_skus.append(effective_sku)
    
    # Create expanded elasticity matrix
    n_original = len(all_original_skus)
    original_elasticity_matrix = np.zeros((n_original, n_original))
    
    # Map elasticities from effective to original SKUs
    for i, orig_sku_i in enumerate(all_original_skus):
        # Find which effective SKU this original SKU maps to
        effective_i = None
        for j, eff_sku in enumerate(effective_skus):
            if eff_sku in reverse_mapping and orig_sku_i in reverse_mapping[eff_sku]:
                effective_i = j
                break
            elif eff_sku == orig_sku_i:
                effective_i = j
                break
        
        for k, orig_sku_k in enumerate(all_original_skus):
            # Find which effective SKU this original SKU maps to
            effective_k = None
            for l, eff_sku in enumerate(effective_skus):
                if eff_sku in reverse_mapping and orig_sku_k in reverse_mapping[eff_sku]:
                    effective_k = l
                    break
                elif eff_sku == orig_sku_k:
                    effective_k = l
                    break
            
            # Copy elasticity from effective matrix
            if effective_i is not None and effective_k is not None:
                original_elasticity_matrix[i, k] = learned_elasticity_matrix[effective_i, effective_k]
    
    return original_elasticity_matrix, all_original_skus

