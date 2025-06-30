use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen]
pub fn preprocess_data( JsValue) -> JsValue {
    let data_obj: HashMap<String, Vec<f64>> = data.into_serde().unwrap();
    let mut features = HashMap::new();
    features.insert("ma_5".to_string(), moving_average(&data_obj["multipliers"], 5));
    features.insert("ma_10".to_string(), moving_average(&data_obj["multipliers"], 10));
    JsValue::from_serde(&features).unwrap()
}

fn moving_average( &[f64], window: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in 0..data.len() {
        if i < window {
            result.push(data[..i+1].iter().sum::<f64>() / (i+1) as f64);
        } else {
            let sum: f64 = data[i-window+1..i+1].iter().sum();
            result.push(sum / window as f64);
        }
    }
    result
}