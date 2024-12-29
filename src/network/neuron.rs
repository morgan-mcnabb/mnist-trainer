

use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    pub raw_value: f32,       
    pub weights: Array1<f32>, 
    pub bias: f32,            
    pub delta: f32,           
    pub activated_value: f32, 
}

impl Neuron {
    pub fn new(num_inputs: usize, activation: &str) -> Self {
        let scale = match activation {
            "sigmoid" => (1.0 / num_inputs as f32).sqrt(),
            _ => 0.5,
        };

        let weights = if num_inputs > 0 {
            let mut rng = rand::thread_rng();
            Array1::random_using(num_inputs, Uniform::new(-scale, scale), &mut rng)
        } else {
            Array1::zeros(0)
        };

        let mut rng = rand::thread_rng();
        let bias = rng.gen_range(-scale..scale);

        Neuron {
            raw_value: 0.0,
            weights,
            bias,
            delta: 0.0,
            activated_value: 0.0,
        }
    }
}

