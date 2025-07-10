use wasm_bindgen::prelude::*;
use std::collections::HashMap;
use std::f64;

// Define feature types for JS interoperability
#[wasm_bindgen]
pub struct Preprocessor {}

#[wasm_bindgen]
impl Preprocessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Preprocessor {}
    }

    // Calculate moving averages
    pub fn moving_average(&self, data: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || window > data.len() {
            return vec![];
        }
        
        let mut result = Vec::with_capacity(data.len() - window + 1);
        let mut sum = data.iter().take(window).sum::<f64>();
        result.push(sum / window as f64);
        
        for i in window..data.len() {
            sum += data[i] - data[i - window];
            result.push(sum / window as f64);
        }
        
        result
    }

    // Calculate momentum indicators
    pub fn momentum(&self, data: &[f64], period: usize) -> Vec<f64> {
        if period == 0 || period >= data.len() {
            return vec![];
        }
        
        data.iter()
            .enumerate()
            .filter_map(|(i, &val)| {
                if i >= period {
                    Some(val - data[i - period])
                } else {
                    None
                }
            })
            .collect()
    }

    // Detect and remove outliers using IQR method
    pub fn remove_outliers(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1_index = sorted.len() / 4;
        let q3_index = (sorted.len() as f64 * 0.75) as usize;
        
        let q1 = sorted[q1_index];
        let q3 = sorted[q3_index];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        data.iter()
            .filter(|&&x| x >= lower_bound && x <= upper_bound)
            .copied()
            .collect()
    }

    // Normalize data between 0 and 1
    pub fn normalize(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }
        
        let min = data.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        let max = data.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        
        if (max - min).abs() < f64::EPSILON {
            return vec![0.5; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - min) / (max - min))
            .collect()
    }

    // Create sequences for LSTM input
    pub fn create_sequences(&self, data: &[f64], seq_length: usize) -> JsValue {
        if seq_length == 0 || seq_length >= data.len() {
            return JsValue::UNDEFINED;
        }
        
        let mut sequences = Vec::new();
        let mut targets = Vec::new();
        
        for i in 0..data.len() - seq_length {
            sequences.push(data[i..i + seq_length].to_vec());
            targets.push(data[i + seq_length]);
        }
        
        // Convert to JS-compatible format
        let js_sequences: Vec<JsValue> = sequences
            .into_iter()
            .map(|seq| JsValue::from_serde(&seq).unwrap())
            .collect();
        
        let js_targets: Vec<JsValue> = targets
            .into_iter()
            .map(JsValue::from)
            .collect();
        
        JsValue::from_serde(&serde_json::json!({
            "sequences": js_sequences,
            "targets": js_targets
        })).unwrap()
    }

    // Add time-based features
    pub fn add_time_features(&self, data: &[f64], timestamp: u64) -> JsValue {
        let now = chrono::NaiveDateTime::from_timestamp_opt(timestamp as i64, 0)
            .unwrap_or(chrono::NaiveDateTime::default());
        
        let hour_of_day = now.hour() as f64 / 24.0;
        let day_of_week = now.weekday().num_days_from_monday() as f64 / 7.0;
        let month_of_year = now.month() as f64 / 12.0;
        
        let features = data.iter()
            .map(|&val| vec![val, hour_of_day, day_of_week, month_of_year])
            .collect::<Vec<_>>();
        
        JsValue::from_serde(&features).unwrap()
    }

    // Augment data with synthetic samples
    pub fn augment_data(&self, data: &[f64], noise_level: f64) -> Vec<f64> {
        if data.is_empty() || noise_level <= 0.0 {
            return data.to_vec();
        }
        
        let mut rng = rand::thread_rng();
        data.iter()
            .map(|&x| {
                x * (1.0 + noise_level * (rand::Rng::gen_range(&mut rng, -1.0..1.0))
            })
            .collect()
    }

    // Calculate volatility index
    pub fn volatility_index(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        variance.sqrt()
    }

    // Create technical indicators
    pub fn technical_indicators(&self, data: &[f64]) -> JsValue {
        let ma_5 = self.moving_average(data, 5);
        let ma_10 = self.moving_average(data, 10);
        let momentum_3 = self.momentum(data, 3);
        
        JsValue::from_serde(&serde_json::json!({
            "ma_5": ma_5,
            "ma_10": ma_10,
            "momentum_3": momentum_3
        })).unwrap()
    }

    // Batch processing for large datasets
    pub fn batch_process(&self, data: &[f64], batch_size: usize) -> JsValue {
        if batch_size == 0 || data.is_empty() {
            return JsValue::UNDEFINED;
        }
        
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        
        for (i, &value) in data.iter().enumerate() {
            current_batch.push(value);
            
            if (i + 1) % batch_size == 0 || i == data.len() - 1 {
                let batch_stats = self.calculate_batch_stats(&current_batch);
                batches.push(batch_stats);
                current_batch.clear();
            }
        }
        
        JsValue::from_serde(&batches).unwrap()
    }

    // Calculate batch statistics
    fn calculate_batch_stats(&self, batch: &[f64]) -> serde_json::Value {
        if batch.is_empty() {
            return serde_json::json!({});
        }
        
        let min = batch.iter().copied().fold(f64::INFINITY, f64::min);
        let max = batch.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = batch.iter().sum::<f64>() / batch.len() as f64;
        
        serde_json::json!({
            "min": min,
            "max": max,
            "mean": mean,
            "size": batch.len(),
            "range": max - min
        })
    }

    // Detect and handle missing values
    pub fn handle_missing_values(&self, data: &[f64], method: &str) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }
        
        match method {
            "forward_fill" => {
                let mut result = Vec::with_capacity(data.len());
                let mut last_valid = data[0];
                
                for &value in data {
                    if value.is_nan() || value.is_infinite() {
                        result.push(last_valid);
                    } else {
                        result.push(value);
                        last_valid = value;
                    }
                }
                
                result
            },
            "backward_fill" => {
                let mut result = data.to_vec();
                let mut next_valid = f64::NAN;
                
                for i in (0..result.len()).rev() {
                    if result[i].is_nan() || result[i].is_infinite() {
                        if !next_valid.is_nan() {
                            result[i] = next_valid;
                        }
                    } else {
                        next_valid = result[i];
                    }
                }
                
                result
            },
            "interpolate" => {
                let mut result = data.to_vec();
                let mut last_valid = f64::NAN;
                let mut last_index = 0;
                
                for (i, &value) in result.iter_mut().enumerate() {
                    if value.is_nan() || value.is_infinite() {
                        continue;
                    }
                    
                    if last_valid.is_nan() {
                        // Fill from start
                        for j in 0..i {
                            result[j] = value;
                        }
                    } else {
                        // Linear interpolation
                        let count = i - last_index;
                        let diff = value - last_valid;
                        
                        for j in last_index..i {
                            result[j] = last_valid + diff * (j - last_index) as f64 / count as f64;
                        }
                    }
                    
                    last_valid = value;
                    last_index = i;
                }
                
                result
            },
            _ => data.to_vec(),
        }
    }

    // Feature importance analysis
    pub fn feature_importance(&self, data: &[f64]) -> JsValue {
        // Simple variance-based importance calculation
        let variance = self.volatility_index(data);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let skewness = self.calculate_skewness(data);
        
        JsValue::from_serde(&serde_json::json!({
            "variance": variance,
            "mean": mean,
            "skewness": skewness,
            "trend": self.calculate_trend(data)
        })).unwrap()
    }

    // Calculate data trend
    fn calculate_trend(&self, data: &[f64]) -> String {
        if data.len() < 2 {
            return "neutral".to_string();
        }
        
        let first = data[0];
        let last = data[data.len() - 1];
        
        if last > first * 1.05 {
            "upward".to_string()
        } else if last < first * 0.95 {
            "downward".to_string()
        } else {
            "neutral".to_string()
        }
    }

    // Calculate data skewness
    fn calculate_skewness(&self, data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std_dev = self.volatility_index(data);
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / data.len() as f64
    }

    // Calculate data drift detection
    pub fn detect_drift(&self, data: &[f64], baseline: &[f64]) -> JsValue {
        if data.is_empty() || baseline.is_empty() {
            return JsValue::from_serde(&serde_json::json!({})).unwrap();
        }
        
        let data_mean = data.iter().sum::<f64>() / data.len() as f64;
        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        
        let drift_score = (data_mean - baseline_mean).abs() / baseline_mean;
        
        JsValue::from_serde(&serde_json::json!({
            "drift_score": drift_score,
            "data_mean": data_mean,
            "baseline_mean": baseline_mean,
            "significant_drift": drift_score > 0.1
        })).unwrap()
    }

    // Create lag features
    pub fn create_lag_features(&self, data: &[f64], lag: usize) -> JsValue {
        if lag == 0 || lag >= data.len() {
            return JsValue::UNDEFINED;
        }
        
        let features = data.windows(lag + 1)
            .map(|window| {
                let target = window[lag];
                let features = &window[..lag];
                serde_json::json!({
                    "features": features,
                    "target": target
                })
            })
            .collect::<Vec<_>>();
        
        JsValue::from_serde(&features).unwrap()
    }

    // Calculate rolling statistics
    pub fn rolling_stats(&self, data: &[f64], window: usize) -> JsValue {
        if window == 0 || window > data.len() {
            return JsValue::UNDEFINED;
        }
        
        let mut stats = Vec::with_capacity(data.len() - window + 1);
        
        for i in 0..data.len() - window + 1 {
            let window_data = &data[i..i + window];
            let sum = window_data.iter().sum::<f64>();
            let mean = sum / window as f64;
            let variance = window_data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            
            stats.push(serde_json::json!({
                "window": i,
                "mean": mean,
                "std_dev": variance.sqrt(),
                "min": window_data.iter().copied().fold(f64::INFINITY, f64::min),
                "max": window_data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            }));
        }
        
        JsValue::from_serde(&stats).unwrap()
    }

    // Encode categorical features
    pub fn encode_categorical(&self, data: &[String]) -> Vec<f64> {
        let mut unique_values = HashMap::new();
        let mut counter = 0;
        
        data.iter()
            .map(|value| {
                *unique_values.entry(value.clone()).or_insert_with(|| {
                    counter += 1;
                    counter - 1
                }) as f64
            })
            .collect()
    }

    // Create polynomial features
    pub fn create_polynomial_features(&self, data: &[f64], degree: u32) -> JsValue {
        let features = data.iter()
            .map(|&x| {
                (0..degree)
                    .map(|d| x.powu(d + 1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        
        JsValue::from_serde(&features).unwrap()
    }

    // Calculate stationarity metrics
    pub fn calculate_stationarity(&self, data: &[f64]) -> JsValue {
        if data.len() < 2 {
            return JsValue::from_serde(&serde_json::json!({})).unwrap();
        }
        
        let mut differences = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            differences.push(data[i] - data[i-1]);
        }
        
        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let var_diff = differences.iter()
            .map(|&x| (x - mean_diff).powi(2))
            .sum::<f64>() / differences.len() as f64;
        
        JsValue::from_serde(&serde_json::json!({
            "mean_diff": mean_diff,
            "var_diff": var_diff,
            "stationary": var_diff < 0.1
        })).unwrap()
    }

    // Calculate seasonality
    pub fn detect_seasonality(&self, data: &[f64], period: usize) -> JsValue {
        if period == 0 || period >= data.len() {
            return JsValue::UNDEFINED;
        }
        
        let mut seasonal_components = Vec::with_capacity(period);
        for i in 0..period {
            let mut values = Vec::new();
            for j in (i..data.len()).step_by(period) {
                values.push(data[j]);
            }
            
            if !values.is_empty() {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                seasonal_components.push(avg);
            }
        }
        
        JsValue::from_serde(&serde_json::json!({
            "period": period,
            "seasonal_components": seasonal_components,
            "strength": seasonal_components.iter().fold(0.0, |acc, &x| acc + x.abs()) 
                / seasonal_components.len() as f64
        })).unwrap()
    }

    // Create lagged features for time series
    pub fn create_lagged_features(&self, data: &[f64], lag: usize) -> JsValue {
        if lag == 0 || lag >= data.len() {
            return JsValue::UNDEFINED;
        }
        
        let mut features = Vec::with_capacity(data.len() - lag);
        for i in lag..data.len() {
            features.push(serde_json::json!({
                "lagged": data[i - lag],
                "current": data[i]
            }));
        }
        
        JsValue::from_serde(&features).unwrap()
    }

    // Detect anomalies
    pub fn detect_anomalies(&self, data: &[f64], threshold: f64) -> JsValue {
        if data.is_empty() {
            return JsValue::UNDEFINED;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std_dev = self.volatility_index(data);
        
        let anomalies: Vec<_> = data.iter()
            .enumerate()
            .filter(|(_, &x)| (x - mean).abs() > threshold * std_dev)
            .map(|(i, &x)| {
                serde_json::json!({
                    "index": i,
                    "value": x,
                    "z_score": (x - mean) / std_dev
                })
            })
            .collect();
        
        JsValue::from_serde(&anomalies).unwrap()
    }

    // Create rolling window features
    pub fn rolling_window(&self, data: &[f64], window: usize) -> JsValue {
        if window == 0 || data.is_empty() {
            return JsValue::UNDEFINED;
        }
        
        let mut windows = Vec::with_capacity(data.len() - window + 1);
        for i in 0..data.len() - window + 1 {
            windows.push(serde_json::json!({
                "start": i,
                "end": i + window,
                "window": &data[i..i + window]
            }));
        }
        
        JsValue::from_serde(&windows).unwrap()
    }

    // Create time-based features
    pub fn create_time_features(&self, timestamps: &[u64]) -> JsValue {
        let features = timestamps.iter()
            .map(|&ts| {
                let dt = chrono::NaiveDateTime::from_timestamp_opt(ts as i64, 0)
                    .unwrap_or(chrono::NaiveDateTime::default());
                
                serde_json::json!({
                    "hour": dt.hour(),
                    "day_of_week": dt.weekday().num_days_from_monday(),
                    "month": dt.month(),
                    "quarter": (dt.month() as f64 / 3.0).ceil() as u32
                })
            })
            .collect::<Vec<_>>();
        
        JsValue::from_serde(&features).unwrap()
    }
}