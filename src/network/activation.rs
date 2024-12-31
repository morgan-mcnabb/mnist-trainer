use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    ReLU,
    Softmax
    // more later
}

impl Activation {
    
    pub fn activate(&self, input: f32) -> f32 {
        match self {
            Activation::Sigmoid => sigmoid(input),
            Activation::ReLU => relu(input),
            Activation::Softmax => input // softmax is apply seperately
        }
    }

    pub fn derivate(&self, input: f32) -> f32 {
        match self {
            Activation::Sigmoid => sigmoid_derivative(input),
            Activation::ReLU => relu_derivative(input),
            Activation::Softmax => 1.0 // this is not used
        }
    }
}
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_derivative(s: f32) -> f32 {
    s * (1.0 - s)
}

fn relu(z: f32) -> f32 {
    z.max(0.0)
}

fn relu_derivative(a: f32) -> f32 {
    if a > 0.0 { 1.0 } else { 0.0 }
}
pub fn softmax(z: &Array1<f32>) -> Vec<f32> {
    let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|&x| (x - max_z).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum_exps).collect()
}

