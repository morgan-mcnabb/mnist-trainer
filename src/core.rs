use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Sample {
    pub inputs: Vec<f32>,
    pub target: Vec<f32>
}

pub trait Layer: Debug {
    fn forward(&mut self, inputs: &[f32]) -> Vec<f32>;
    fn backward(&mut self, deltas: &[f32]) -> Vec<f32>;
    fn update_weights(&mut self, learning_rate: f32);
}

pub trait Trainable {
    fn forward_pass(&mut self, inputs: &[f32]) -> Vec<f32>;
    fn back_propagate(&mut self, target: &[f32], learning_rate: f32);
    fn calculate_loss(&self, targets: &[f32]) -> f32;
}
