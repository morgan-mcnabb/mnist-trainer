
// src/training.rs

use crate::core::{Trainable, Sample};
use crate::dataset::Dataset;
use crate::neural_network::mlp::MLP;

pub fn train_model(
    network: &mut impl Trainable,
    data: &mut [Sample],
    epochs: usize,
    learning_rate: f32,
) {
    for epoch in 0..epochs {
        // shuffle data, etc.
        // forward_pass + back_propagate in a loop
        // compute & print loss or accuracy
    }
}
