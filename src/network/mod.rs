
pub mod neuron;
pub mod layer;
pub mod activation;

use crate::network::layer::Layer;

pub fn initialize_network(layer_sizes: &[usize]) -> Vec<Layer> {
    let mut layers = Vec::new();

    // Input layer: no incoming weights
    layers.push(Layer::new(layer_sizes[0], 0, "none"));

    // Hidden and output layers
    for i in 1..layer_sizes.len() {
        let activation = if i == layer_sizes.len() - 1 {
            "softmax"
        } else {
            "sigmoid"
        };
        layers.push(Layer::new(layer_sizes[i], layer_sizes[i - 1], activation));
    }

    layers
}

