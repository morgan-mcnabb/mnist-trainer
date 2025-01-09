use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub epochs: usize,
    pub learning_rate: f32,
    pub layers: Vec<usize>,
    pub activations: Vec<String>,
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            epochs: 20,
            learning_rate: 0.1,
            layers: vec![784, 128, 64, 10],
            activations: vec!["sigmoid".to_string(), "sigmoid".to_string(), "sigmoid".to_string()],
            batch_size: 32,
        }
    }
}
