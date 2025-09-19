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