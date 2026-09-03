import numpy as np
import pandas as pd
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

class SoilMoistureForecaster:
    """
    Accelerated Forecaster Engine for Volumetric Soil Moisture (v2).
    Optimized for multi-core CPUs (AMD Ryzen 9 5950X) and NVIDIA GPUs.
    Supports single-location recursive forecasting and vectorized multi-location batch forecasting.
    """
    
    def __init__(self, model, scaler_X, scaler_y, look_back=14):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.look_back = look_back
        
        # Features: [Precipitation, T_Max, T_Min, SMAP_Moisture]
        self.feature_names = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']

    def _fast_predict(self, x_input):
        """Executes direct model call to bypass Keras graph construction overhead per step."""
        try:
            return self.model(x_input, training=False).numpy()
        except Exception:
            return self.model.predict(x_input, verbose=0)

    def predict_future(self, recent_data, forecast_horizon=14, future_meteorological_forcing=None):
        """Single-location recursive multi-step forecasting."""
        if len(recent_data) != self.look_back:
            raise ValueError(f"recent_data must contain exactly {self.look_back} days.")
            
        if isinstance(recent_data, pd.DataFrame):
            current_sequence = recent_data[self.feature_names].values.copy()
        else:
            current_sequence = np.array(recent_data, dtype=np.float32).copy()
            
        current_sequence_scaled = self.scaler_X.transform(current_sequence)
        forecasted_swc = []
        
        for step in range(forecast_horizon):
            lstm_input = current_sequence_scaled.reshape(1, self.look_back, len(self.feature_names))
            pred_swc_scaled = self._fast_predict(lstm_input)
            pred_swc_actual = self.scaler_y.inverse_transform(pred_swc_scaled.reshape(-1, 1))[0, 0]
            forecasted_swc.append(pred_swc_actual)
            
            next_day_features = np.zeros((1, len(self.feature_names)))
            
            if future_meteorological_forcing is not None and step < len(future_meteorological_forcing):
                next_day_features[0, 0:3] = future_meteorological_forcing[step, 0:3]
            else:
                next_day_features[0, 0] = 0.0  # Precip = 0
                next_day_features[0, 1] = current_sequence[-1, 1]  # T_Max
                next_day_features[0, 2] = current_sequence[-1, 2]  # T_Min
            
            next_day_features[0, 3] = pred_swc_actual  # Use prediction as new state
            
            next_day_scaled = self.scaler_X.transform(next_day_features)
            current_sequence_scaled = np.vstack([current_sequence_scaled[1:], next_day_scaled])
            current_sequence = np.vstack([current_sequence[1:], next_day_features])

        return np.array(forecasted_swc)

    def predict_future_batch(self, recent_batch, forecast_horizon=14, future_forcing_batch=None):
        """
        Vectorized Multi-Location Batch Forecasting (v2 Accelerated).
        Vectorizes scaling and tensor operations across all N spatial locations simultaneously.
        
        recent_batch: shape (N, look_back, 4)
        future_forcing_batch: shape (N, forecast_horizon, 3) or (forecast_horizon, 3)
        returns: shape (N, forecast_horizon)
        """
        recent_arr = np.array(recent_batch, dtype=np.float32)
        N = recent_arr.shape[0]
        
        # Single-pass 2D scaling across all N locations simultaneously
        reshaped_2d = recent_arr.reshape(-1, 4)
        scaled_2d = self.scaler_X.transform(reshaped_2d)
        curr_seq_scaled = scaled_2d.reshape(N, self.look_back, 4)
            
        forecasts = np.zeros((N, forecast_horizon), dtype=np.float32)
        
        for step in range(forecast_horizon):
            # Batch tensor inference across all N locations in one GPU/CPU operation
            preds_scaled = self._fast_predict(curr_seq_scaled)  # shape (N, 1)
            preds_actual = self.scaler_y.inverse_transform(preds_scaled).flatten()  # shape (N,)
            forecasts[:, step] = preds_actual
            
            # Construct next-day feature vectors for all N locations in a single matrix
            next_day = np.zeros((N, 4), dtype=np.float32)
            
            if future_forcing_batch is not None:
                if future_forcing_batch.ndim == 3:  # (N, horizon, 3)
                    next_day[:, 0:3] = future_forcing_batch[:, step, 0:3]
                elif future_forcing_batch.ndim == 2:  # (horizon, 3)
                    next_day[:, 0:3] = future_forcing_batch[step, 0:3]
            else:
                next_day[:, 0] = 0.0  # Precip = 0
                next_day[:, 1] = curr_seq_scaled[:, -1, 1]
                next_day[:, 2] = curr_seq_scaled[:, -1, 2]
                
            next_day[:, 3] = preds_actual
            
            # Vectorized scaling of next-day vectors for all N locations at once
            next_day_scaled = self.scaler_X.transform(next_day)
                
            # Shift rolling window: drop oldest day, append new day for all N points
            curr_seq_scaled = np.concatenate(
                [curr_seq_scaled[:, 1:, :], next_day_scaled[:, np.newaxis, :]], 
                axis=1
            )

        return forecasts
