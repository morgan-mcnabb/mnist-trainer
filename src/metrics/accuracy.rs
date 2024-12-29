
use crate::network::layer::Layer;
use crate::data::dataset::Sample;
use ndarray::Array1;
use crate::training::trainer::forward_pass;

fn argmax(vals: &Array1<f32>) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Evaluates the accuracy of the network on a given dataset.
pub fn evaluate(layers: &mut [Layer], dataset: &[Sample]) -> f32 {
    let out_idx = layers.len() - 1;
    let mut correct = 0;
    for sample in dataset {
        forward_pass(layers, &sample.inputs);
        let prediction = argmax(&layers[out_idx].activated_values());
        let actual = argmax(&sample.target);
        if prediction == actual {
            correct += 1;
        }
    }
    (correct as f32 / dataset.len() as f32) * 100.0
}


