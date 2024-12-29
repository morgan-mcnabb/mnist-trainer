use crate::network::neuron::Neuron;
use ndarray::Array1;

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs: usize, activation: &str) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(num_inputs, activation))
            .collect();
        Layer { neurons }
    }

    pub fn activated_values(&self) -> Array1<f32> {
        Array1::from_iter(self.neurons.iter().map(|n| n.activated_value))
    }
}

