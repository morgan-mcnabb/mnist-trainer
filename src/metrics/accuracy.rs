use crate::network::layer::Layer;
use crate::data::dataset::Sample;

/// Returns the index of the maximum value in a slice.
fn argmax(vals: &[f32]) -> usize {
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
        // Forward pass
        // Note: You might want to abstract forward_pass to be accessible here
        // For simplicity, assuming it's accessible via training::trainer::forward_pass
        crate::training::trainer::forward_pass(layers, &sample.inputs);

        // Prediction
        let prediction = argmax(
            &layers[out_idx]
                .neurons
                .iter()
                .map(|n| n.activated_value)
                .collect::<Vec<f32>>(),
        );
        let actual = argmax(&sample.target);

        if prediction == actual {
            correct += 1;
        }
    }
    (correct as f32 / dataset.len() as f32) * 100.0
}

