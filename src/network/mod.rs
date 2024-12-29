
pub mod neuron;
pub mod layer;
pub mod activation;

use crate::network::activation::Activation;
use crate::network::layer::Layer;

pub fn initialize_network(layer_sizes: &[usize], activations: &[Activation]) -> Vec<Layer> {
    let mut layers = Vec::new();

    // input layer
    layers.push(Layer::new(layer_sizes[0], 0, None));

    // hidden layers
    for i in 1..layer_sizes.len() - 1 {
        layers.push(Layer::new(layer_sizes[i], layer_sizes[i - 1], Some(activations[i - 1])));
    }

    // output layer
    layers.push(Layer::new(layer_sizes[layer_sizes.len() - 1], layer_sizes[layer_sizes.len() - 2], Some(Activation::Softmax)));

    layers
}

