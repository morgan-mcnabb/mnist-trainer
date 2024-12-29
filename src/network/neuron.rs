use rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct Neuron {
    pub raw_value: f32,       // Weighted sum (z)
    pub weights: Vec<f32>,    // Incoming weights
    pub bias: f32,            // Bias
    pub delta: f32,           // Delta for backprop
    pub activated_value: f32, // Activated output (a)
}

impl Neuron {
    /// Creates a new Neuron with random weights and biases.
    pub fn new(num_inputs: usize, activation: &str) -> Self {
        let mut rng = thread_rng();
        let scale = match activation {
            "sigmoid" => (1.0 / num_inputs as f32).sqrt(),
            _ => 0.5,
        };

        let weights: Vec<f32> = if num_inputs > 0 {
            (0..num_inputs).map(|_| rng.gen_range(-scale..scale)).collect()
        } else {
            Vec::new()
        };

        Neuron {
            raw_value: 0.0,
            weights,
            bias: rng.gen_range(-scale..scale),
            delta: 0.0,
            activated_value: 0.0,
        }
    }
}

