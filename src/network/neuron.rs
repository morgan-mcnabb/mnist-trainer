use crate::network::activation::Activation;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub raw_value: f32,       
    pub weights: Array1<f32>, 
    pub bias: f32,            
    pub delta: f32,           
    pub activated_value: f32, 
}

impl Neuron {
    pub fn new(num_inputs: usize, activation: Option<&Activation>) -> Self {
         let (weights, bias) = if num_inputs > 0 {
            let activation = activation.expect("Activation must be provided for non-input layers.");
            let scale = match activation {
                Activation::Sigmoid | Activation::Softmax => (1.0 / num_inputs as f32).sqrt(),
                Activation::ReLU => (2.0 / num_inputs as f32).sqrt(),
            };

            let weights = {
                let mut rng = rand::thread_rng();
                Array1::random_using(num_inputs, Uniform::new(-scale, scale), &mut rng)
            };

            let bias = {
                let mut rng = rand::thread_rng();
                rng.gen_range(-scale..scale)
            };

            (weights, bias)
        } else {
            (Array1::zeros(0), 0.0)
        };
        Neuron {
            raw_value: 0.0,
            weights,
            bias,
            delta: 0.0,
            activated_value: 0.0,
        }
    }
}

