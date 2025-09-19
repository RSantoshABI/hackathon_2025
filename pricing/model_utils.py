import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict


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

def calculate_metrics(model, data: Dict, data_type: str = "train") -> Dict[str, float]:
    """
    Calculate R² and MAPE metrics for the elasticity model
    
    Args:
        model: Trained ElasticityModel instance
        data: Dictionary with 'log_prices', 'features', 'log_volumes'
        data_type: String identifier for the dataset ("train", "validation", etc.)
        
    Returns:
        Dictionary with calculated metrics
    """
    
    # Get predictions from model
    predictions = model.predict(
        tf.constant(data['log_prices'], dtype=tf.float32),
        tf.constant(data['features'], dtype=tf.float32)
    )
    
    # Convert to numpy arrays - THIS IS THE KEY FIX
    if isinstance(data['log_volumes'], tf.Tensor):
        y_true = data['log_volumes'].numpy()
    else:
        y_true = np.array(data['log_volumes'])
    
    if isinstance(predictions, tf.Tensor):
        y_pred = predictions.numpy()
    else:
        y_pred = np.array(predictions)
    
    # Convert back to actual volumes for MAPE calculation
    actual_volumes = np.exp(y_true)
    predicted_volumes = np.exp(y_pred)
    
    # Calculate R² (coefficient of determination)
    r2_scores = []
    
    # Calculate R² for each SKU separately
    for sku_idx in range(y_true.shape[1]):
        sku_true = y_true[:, sku_idx]
        sku_pred = y_pred[:, sku_idx]
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((sku_true - sku_pred) ** 2)
        ss_tot = np.sum((sku_true - np.mean(sku_true)) ** 2)
        
        if ss_tot == 0:  # Avoid division by zero
            r2_sku = 0.0
        else:
            r2_sku = 1 - (ss_res / ss_tot)
        
        r2_scores.append(r2_sku)
    
    # Overall R² (weighted by total variance)
    overall_r2 = calculate_overall_r2(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape_scores = []
    
    # Calculate MAPE for each SKU separately
    for sku_idx in range(actual_volumes.shape[1]):
        sku_actual = actual_volumes[:, sku_idx]
        sku_predicted = predicted_volumes[:, sku_idx]
        
        # Avoid division by zero in MAPE calculation
        non_zero_mask = sku_actual > 0.01  # Only consider non-zero actual values
        
        if np.sum(non_zero_mask) > 0:
            mape_sku = np.mean(
                np.abs((sku_actual[non_zero_mask] - sku_predicted[non_zero_mask]) / sku_actual[non_zero_mask])
            ) * 100
        else:
            mape_sku = 0.0
        
        mape_scores.append(mape_sku)
    
    # Overall MAPE
    overall_mape = calculate_overall_mape(actual_volumes, predicted_volumes)
    
    # Additional metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        f'{data_type}_r2_overall': overall_r2,
        f'{data_type}_r2_mean': np.mean(r2_scores),
        f'{data_type}_r2_median': np.median(r2_scores),
        f'{data_type}_r2_min': np.min(r2_scores),
        f'{data_type}_r2_max': np.max(r2_scores),
        f'{data_type}_mape_overall': overall_mape,
        f'{data_type}_mape_mean': np.mean(mape_scores),
        f'{data_type}_mape_median': np.median(mape_scores),
        f'{data_type}_mape_min': np.min(mape_scores),
        f'{data_type}_mape_max': np.max(mape_scores),
        f'{data_type}_rmse': rmse,
        f'{data_type}_mae': mae,
        f'{data_type}_num_skus': len(r2_scores)
    }

def calculate_overall_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate overall R² across all SKUs and time periods"""
    # Ensure inputs are numpy arrays
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Flatten arrays to treat as single prediction problem
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    else:
        return 1 - (ss_res / ss_tot)

def calculate_overall_mape(actual_volumes: np.ndarray, predicted_volumes: np.ndarray) -> float:
    """Calculate overall MAPE across all SKUs and time periods"""
    # Ensure inputs are numpy arrays
    if isinstance(actual_volumes, tf.Tensor):
        actual_volumes = actual_volumes.numpy()
    if isinstance(predicted_volumes, tf.Tensor):
        predicted_volumes = predicted_volumes.numpy()
    
    # Flatten arrays
    actual_flat = actual_volumes.flatten()
    predicted_flat = predicted_volumes.flatten()
    
    # Only consider non-zero actual values
    non_zero_mask = actual_flat > 0.01
    
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(
            np.abs((actual_flat[non_zero_mask] - predicted_flat[non_zero_mask]) / actual_flat[non_zero_mask])
        ) * 100
        return mape
    else:
        return 0.0

def print_metrics_summary(metrics: Dict[str, float], data_type: str = ""):
    """Pretty print metrics summary"""
    print(f"\n=== {data_type.upper()} METRICS SUMMARY ===")
    print(f"R² Scores:")
    print(f"  Overall R²: {metrics.get(f'{data_type}_r2_overall', 0):.4f}")
    print(f"  Mean R²: {metrics.get(f'{data_type}_r2_mean', 0):.4f}")
    print(f"  Median R²: {metrics.get(f'{data_type}_r2_median', 0):.4f}")
    print(f"  Min R²: {metrics.get(f'{data_type}_r2_min', 0):.4f}")
    print(f"  Max R²: {metrics.get(f'{data_type}_r2_max', 0):.4f}")
    
    print(f"\nMAPE Scores:")
    print(f"  Overall MAPE: {metrics.get(f'{data_type}_mape_overall', 0):.2f}%")
    print(f"  Mean MAPE: {metrics.get(f'{data_type}_mape_mean', 0):.2f}%")
    print(f"  Median MAPE: {metrics.get(f'{data_type}_mape_median', 0):.2f}%")
    print(f"  Min MAPE: {metrics.get(f'{data_type}_mape_min', 0):.2f}%")
    print(f"  Max MAPE: {metrics.get(f'{data_type}_mape_max', 0):.2f}%")
    
    print(f"\nOther Metrics:")
    print(f"  RMSE (log scale): {metrics.get(f'{data_type}_rmse', 0):.4f}")
    print(f"  MAE (log scale): {metrics.get(f'{data_type}_mae', 0):.4f}")
    print(f"  Number of SKUs: {metrics.get(f'{data_type}_num_skus', 0)}")