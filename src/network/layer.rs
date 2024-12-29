use crate::network::neuron::Neuron;

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a new Layer with `num_neurons` neurons, each with `num_inputs` incoming weights.
    pub fn new(num_neurons: usize, num_inputs: usize, activation: &str) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(num_inputs, activation))
            .collect();
        Layer { neurons }
    }
}

