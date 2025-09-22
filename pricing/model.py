import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from model_utils import calculate_metrics


class ElasticityModel:
    def __init__(
        self,
        n_skus: int,
        n_features: int,
        learning_rate: float = 0.001,
        reg_lambda: float = 0.01,
    ):
        self.n_skus = n_skus
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        self._build_model()

    def _build_model(self):

        self.elasticity_matrix = tf.Variable(
            tf.random.normal([self.n_skus, self.n_skus], stddev=0.1),
            name="elasticity_matrix",
        )

        self.intercepts = tf.Variable(
            tf.random.normal([self.n_skus], stddev=0.1),
            name="intercepts",
        )

        self.feature_coeffs = tf.Variable(
            tf.random.normal([self.n_skus, self.n_features], stddev=0.1),
            name="feature_coeffs",
        )

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def predict(self, log_prices: tf.Tensor, features: tf.Tensor) -> tf.Tensor:

        price_impact = tf.reduce_sum(
            log_prices[:, None, :] * self.elasticity_matrix[None, :, :], axis=2
        )

        feature_impact = tf.matmul(features, self.feature_coeffs, transpose_b=True)

        predictions = self.intercepts[None, :] + price_impact + feature_impact

        return predictions

    def calculate_loss(
        self, log_prices: tf.Tensor, features: tf.Tensor, true_log_volumes: tf.Tensor
    ) -> Dict[str, tf.Tensor]:

        predictions = self.predict(log_prices, features)

        mse_loss = tf.reduce_mean(tf.square(predictions - true_log_volumes))
        ridge_loss = self.reg_lambda * (  
            tf.reduce_sum(tf.square(self.elasticity_matrix))
            + tf.reduce_sum(tf.square(self.feature_coeffs))
        )
        constraints_loss = self._elasticity_constraint_loss()  

        total_loss = mse_loss + ridge_loss + constraints_loss

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "ridge_loss": ridge_loss,
            "constraint_loss": constraints_loss,
        }

    def _elasticity_constraint_loss(
        self,
        penalty_weight: float = 50.0,
    ) -> tf.Tensor:

        own_elasticities = tf.linalg.diag_part(self.elasticity_matrix)
        cross_elasticities = self.elasticity_matrix - tf.linalg.diag(own_elasticities)

        positive_own_penalty = tf.reduce_sum(tf.nn.relu(own_elasticities))
        extreme_negative_penalty = tf.reduce_sum(tf.nn.relu(-5.0 - own_elasticities))

        negative_cross_penalty = tf.reduce_sum(tf.nn.relu(-cross_elasticities))
        extreme_positive_penalty = tf.reduce_sum(tf.nn.relu(cross_elasticities - 5.0))

        total_constraint_loss = penalty_weight * (
            positive_own_penalty
            + extreme_negative_penalty
            + negative_cross_penalty
            + extreme_positive_penalty
        )

        return total_constraint_loss

    @tf.function
    def _step(
        self, log_prices: tf.Tensor, features: tf.Tensor, true_log_volumes: tf.Tensor
    ) -> Dict[str, tf.Tensor]:

        with tf.GradientTape() as tape:
            losses = self.calculate_loss(log_prices, features, true_log_volumes)
            total_loss = losses["total_loss"]  

        gradients = tape.gradient(
            total_loss,
            [
                self.elasticity_matrix,
                self.intercepts,
                self.feature_coeffs,
            ],
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                [self.elasticity_matrix, self.intercepts, self.feature_coeffs],
            )
        )

        return losses

    def train(
        self, train_data: Dict, epochs: int = 1000, validation_data: Dict = None
        ) -> Dict:

        history = {
            "train_loss": [], 
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }

        for epoch in range(epochs):
            losses = self._step(
                train_data["log_prices"],
                train_data["features"],
                train_data["log_volumes"],
            )

            # CHECK FOR INF/NAN
            current_loss = losses["total_loss"].numpy()
            if np.isnan(current_loss) or np.isinf(current_loss):
                print(f"WARNING: Invalid loss at epoch {epoch}: {current_loss}")
                print(f"MSE: {losses['mse_loss'].numpy()}")
                print(f"Ridge: {losses['ridge_loss'].numpy()}")
                print(f"Constraint: {losses['constraint_loss'].numpy()}")
                break

            history["train_loss"].append(current_loss)

            # Calculate validation loss
            if validation_data is not None:
                val_losses = self.calculate_loss(
                    validation_data["log_prices"],
                    validation_data["features"],
                    validation_data["log_volumes"],
                )
                history["val_loss"].append(val_losses["total_loss"].numpy())

            # Calculate detailed metrics every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {losses['total_loss']:.4f}")
                
                # Calculate training metrics
                train_metrics = calculate_metrics(self, train_data, "train")
                history["train_metrics"].append({
                    'epoch': epoch,
                    **train_metrics
                })
                
                print(f"  Train R²: {train_metrics['train_r2_overall']:.4f}, Train MAPE: {train_metrics['train_mape_overall']:.2f}%")
                
                if validation_data is not None:
                    val_metrics = calculate_metrics(self, validation_data, "val")
                    history["val_metrics"].append({
                        'epoch': epoch,
                        **val_metrics
                    })
                    print(f"  Val Loss = {val_losses['total_loss']:.4f}")
                    print(f"  Val R²: {val_metrics['val_r2_overall']:.4f}, Val MAPE: {val_metrics['val_mape_overall']:.2f}%")

        return history
        
    def get_elasticity_matrix(self) -> np.ndarray:
        return self.elasticity_matrix.numpy()

    # EXPERIMENTAL DO NOT RUN
    def predict_test_volumes(
            self,
            test_data: pd.DataFrame,
            train_skus: list,
            price_matrix,
            feature_matrix) -> pd.DataFrame:
        
        results_df = test_data.copy()
        
        test_skus = set(test_data['sku'].unique())
        train_skus_set = set(train_skus)
        
        known_skus = test_skus.intersection(train_skus_set)
        new_skus = test_skus - train_skus_set
        
        print(f"SKU Analysis:")
        print(f"  Total test SKUs: {len(test_skus)}")
        print(f"  Known SKUs (will predict): {len(known_skus)}")
        print(f"  New SKUs (will set to 0): {len(new_skus)}")
        
        results_df['predicted_volume'] = 0.0
        
        if known_skus:
            known_data = test_data[test_data['sku'].isin(known_skus)].copy()
            predicted_volumes = self._predict_for_known_skus(known_data, price_matrix, feature_matrix)
            
            for idx, row in known_data.iterrows():
                sku = row['sku']
                date = row['date']
                
                pred_idx = ((known_data['sku'] == sku) & (known_data['date'] == date)).idxmax()
                prediction = predicted_volumes[pred_idx] if pred_idx in predicted_volumes.index else 0.0
                
                mask = (results_df['sku'] == sku) & (results_df['date'] == date)
                results_df.loc[mask, 'predicted_volume'] = prediction
        
        total_predictions = len(results_df)
        zero_predictions = (results_df['predicted_volume'] == 0).sum()
        non_zero_predictions = total_predictions - zero_predictions
        
        print(f"\nPrediction Summary:")
        print(f"  Total rows: {total_predictions:,}")
        print(f"  Non-zero predictions: {non_zero_predictions:,}")
        print(f"  Zero predictions: {zero_predictions:,}")
        
        return results_df


    def _predict_for_known_skus(self, known_data: pd.DataFrame, price_matrix, feature_matrix) -> pd.Series:
        """Quick fix for shape mismatch"""
        expected_n_skus = self.elasticity_matrix.shape[0]
        actual_n_skus = price_matrix.shape[1]
        
        print(f"Expected SKUs: {expected_n_skus}, Actual SKUs: {actual_n_skus}")
        
        if actual_n_skus != expected_n_skus:
            print(f"Shape mismatch detected. Padding/truncating price matrix...")
            
            if actual_n_skus < expected_n_skus:
                padding_cols = expected_n_skus - actual_n_skus
                avg_prices = price_matrix.mean(axis=1)
                
                padding_data = pd.DataFrame(
                    np.tile(avg_prices.values.reshape(-1, 1), (1, padding_cols)),
                    index=price_matrix.index,
                    columns=[f'padding_sku_{i}' for i in range(padding_cols)]
                )
                
                aligned_price_matrix = pd.concat([price_matrix, padding_data], axis=1)
                
            else:
                aligned_price_matrix = price_matrix.iloc[:, :expected_n_skus]
        else:
            aligned_price_matrix = price_matrix
        
        log_prices_tensor = tf.constant(aligned_price_matrix.values, dtype=tf.float32)
        features_tensor = tf.constant(feature_matrix, dtype=tf.float32)
        
        predicted_log_volumes = self.predict(log_prices_tensor, features_tensor)
        predicted_volumes = tf.exp(predicted_log_volumes).numpy()
        
        predictions_dict = {}
        original_skus = list(price_matrix.columns)
        
        for date_idx, date in enumerate(price_matrix.index):
            for sku_idx, sku in enumerate(original_skus):
                key = (date, sku)
                predictions_dict[key] = predicted_volumes[date_idx, sku_idx]
        
        predictions = pd.Series(index=known_data.index, dtype=float)
        
        for idx, row in known_data.iterrows():
            key = (row['date'], row['sku'])
            predictions[idx] = predictions_dict.get(key, 0.0)
        
        return predictions