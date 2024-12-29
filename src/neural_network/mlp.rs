
use std::fmt::{Debug, Formatter, Result as FmtResult};
use crate::core::{Layer, Trainable};
use crate::activations::{sigmoid, sigmoid_derivative, softmax};
use crate::losses::cross_entropy_loss;

/// A single neuron layer in an MLP.
#[derive(Debug)]
pub struct MLP_Layer {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub outputs: Vec<f32>,
    pub deltas: Vec<f32>,
    pub is_output: bool,
}

impl MLP_Layer {
    pub fn new(num_neurons: usize, num_inputs: usize, is_output: bool) -> Self {
        use rand::{thread_rng, Rng};

        let mut rng = thread_rng(); 
        let scale = (2.0 / (num_inputs as f32 + num_neurons as f32)).sqrt();

        let mut weights = vec![vec![0.0; num_inputs]; num_neurons];
        for neuron_index in 0..num_neurons {
            for input_index in 0..num_inputs {
                weights[neuron_index][input_index] = rng.gen_range(-scale..scale);
            }
        }

        let mut biases = vec![0.0; num_neurons];
        for bias_index in 0..num_neurons {
            biases[bias_index] = rng.gen_range(-scale..scale);
        }

        MLP_Layer {
            weights,
            biases,
            outputs: vec![0.0; num_neurons],
            deltas: vec![0.0; num_neurons],
            is_output,
        }
    }
    // Initialization, forward, backward, etc.
}

/// Implementation of the `Layer` trait for MLP_Layer
impl Layer for MLP_Layer {
    fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Weighted sums + activation (sigmoid or linear if output)
        // ...
        //
        for (i, neuron_weights) in self.weights.iter().enumerate() {
            let mut z = 0.0;
            for (weight_index, &weight) in neuron_weights.iter().enumerate() {
                z += weight * inputs[weight_index];
            }

            z += self.biases[i];

            if !self.is_output {
                let activated = sigmoid(z);
                self.outputs[i] = activated;
            } else {
                // this is the output layer, store the raw z (might softmax)
                self.outputs[i] = z;
            }
        }

        self.outputs.clone()
    }

    fn backward(&mut self, incoming_deltas: &[f32]) -> Vec<f32> {
        // compute gradients, store self.deltas
        // ...
        //
        //
        let num_neurons = self.outputs.len();
        let num_inputs = self.weights[0].len(); // each neuron will have this many inputs
                                                //
        for i in 0..num_neurons {
            if !self.is_output {
                let raw_z = self.outputs[i];

                let derivative = sigmoid_derivative(raw_z);
                self.deltas[i] = incoming_deltas[i] * derivative;
            } else {
                self.deltas[i] = incoming_deltas[i];
            }
        }

        let mut previous_layer_errors = vec![0.0; num_inputs];
        for i in 0..num_neurons {
            for j in 0..num_inputs {
                previous_layer_errors[j] += self.deltas[i] * self.weights[i][j];
            }
        }
        previous_layer_errors
    }

    fn update_weights(&mut self, learning_rate: f32) {
        // apply gradient to weights
        // ...

        i
    }
}

/// A multi-layer perceptron (MLP) network
#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<MLP_Layer>,
}

impl MLP {
    pub fn build(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();

        for (index, &size) in layer_sizes.iter().enumerate() {
            if index == 0 {
                // input layer, skip
                continue;
            }

            let previous_size = layer_sizes[index - 1];
            let is_output = index == layer_sizes.len() - 1;

            let layer = MLP_Layer::new(size, previous_size, is_output);
            layers.push(layer);
        }

        MLP {layers}
    }
}

impl Trainable for MLP {
    fn forward_pass(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Pass through each MLP_Layer in self.layers

        let mut current_activations = inputs.to_vec();
        
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            current_activations = layer.forward(&current_activations);
        }
        
        current_activations
    }

    fn back_propagate(&mut self, targets: &[f32], learning_rate: f32) {
        // compute deltas at output,
        // pass them backward through layers

        let final_outputs = self.layers.last().unwrap().outputs.clone();
        let mut incoming_deltas = Vec::with_capacity(final_outputs.len());
        for i in 0..final_outputs.len() {
            incoming_deltas.push(final_outputs[i] - targets[i]);
        }

        for layer_index in (0..self.layers.len()).rev() {
            incoming_deltas = self.layers[layer_index].backward(&incoming_deltas);
        }

        for layer_index in 0..self.layers.len() {
            self.layers[layer_index].update_weights(learning_rate);
        }
    }

    fn calculate_loss(&self, targets: &[f32]) -> f32 {
        // cross_entropy_loss(...) or other
        //
        let final_outputs = self.layers.last().unwrap().outputs.as_slice();
        cross_entropy_loss(final_outputs, targets)
    }
}

// Optionally, provide a helper function to build an MLP from a slice of layer sizes
pub fn build_mlp(layer_sizes: &[usize]) -> MLP {
    // Create MLP_Layer for each pair in layer_sizes
    unimplemented!()
}
