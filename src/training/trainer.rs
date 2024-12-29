use crate::network::layer::Layer;
use crate::network::activation;
use crate::data::dataset::Sample;
use crate::utils::math::shuffle_dataset;
use crate::metrics::accuracy::evaluate;

/// Performs a forward pass through the network.
fn forward_pass(layers: &mut [Layer], inputs: &[f32]) {
    let total_layers = layers.len();

    // Input layer activations
    for (i, neuron) in layers[0].neurons.iter_mut().enumerate() {
        neuron.raw_value = inputs[i];
        neuron.activated_value = inputs[i];
    }

    // Hidden and output layers
    for l in 1..total_layers {
        let prev_activations: Vec<f32> = layers[l - 1]
            .neurons
            .iter()
            .map(|n| n.activated_value)
            .collect();

        let is_not_output = l < (total_layers - 1);

        for neuron in &mut layers[l].neurons {
            let weighted_sum: f32 = neuron
                .weights
                .iter()
                .zip(prev_activations.iter())
                .map(|(w, a)| w * a)
                .sum::<f32>()
                + neuron.bias;

            neuron.raw_value = weighted_sum;
            neuron.activated_value = if is_not_output {
                activation::sigmoid(weighted_sum)
            } else {
                weighted_sum // Softmax will be applied later
            };
        }
    }

    // Apply softmax to the output layer
    let output_idx = total_layers - 1;
    let raw_outputs: Vec<f32> = layers[output_idx]
        .neurons
        .iter()
        .map(|n| n.raw_value)
        .collect();

    let softmax_vals = activation::softmax(&raw_outputs);

    for (neuron, &val) in layers[output_idx].neurons.iter_mut().zip(softmax_vals.iter()) {
        neuron.activated_value = val;
    }
}

/// Performs backpropagation and updates weights and biases.
fn back_propagate(layers: &mut [Layer], targets: &[f32], lr: f32) {
    let out_idx = layers.len() - 1;

    // 1. Calculate deltas for output layer
    for (i, neuron) in layers[out_idx].neurons.iter_mut().enumerate() {
        neuron.delta = neuron.activated_value - targets[i];
    }

    // 2. Calculate deltas for hidden layers
    for l in (1..out_idx).rev() {
        for (j, neuron) in layers[l].neurons.iter_mut().enumerate() {
            let sum: f32 = layers[l + 1]
                .neurons
                .iter()
                .map(|n| n.delta * n.weights[j])
                .sum();
            neuron.delta = sum * activation::sigmoid_derivative(neuron.raw_value);
        }
    }

    // 3. Update weights and biases
    for l in 1..layers.len() {
        let prev_activations: Vec<f32> = layers[l - 1]
            .neurons
            .iter()
            .map(|n| n.activated_value)
            .collect();

        for neuron in &mut layers[l].neurons {
            // Update bias
            neuron.bias -= lr * neuron.delta;

            // Update weights
            for (k, weight) in neuron.weights.iter_mut().enumerate() {
                *weight -= lr * neuron.delta * prev_activations[k];
            }
        }
    }
}

/// Calculates cross-entropy loss for the current output.
fn calculate_loss(layers: &[Layer], targets: &[f32]) -> f32 {
    let out_idx = layers.len() - 1;
    layers[out_idx]
        .neurons
        .iter()
        .zip(targets.iter())
        .map(|(n, &t)| -t * (n.activated_value + 1e-12).ln())
        .sum()
}

/// The main training loop.
pub fn train(
    layers: &mut [Layer],
    training_set: &[Sample],
    epochs: usize,
    learning_rate: f32,
    test_set: &[Sample],
) {
    for epoch in 0..epochs {
        // Shuffle the training data
        let mut shuffled = training_set.to_vec();
        shuffle_dataset(&mut shuffled);

        let mut total_loss = 0.0;
        for sample in shuffled.iter() {
            forward_pass(layers, &sample.inputs);
            back_propagate(layers, &sample.target, learning_rate);
            total_loss += calculate_loss(layers, &sample.target);
        }

        // Evaluate accuracy
        let train_acc = evaluate(layers, training_set);
        let test_acc = evaluate(layers, test_set);

        println!(
            "Epoch {}: Loss = {:.4}, Train Acc = {:.2}%, Test Acc = {:.2}%",
            epoch + 1,
            total_loss / training_set.len() as f32,
            train_acc,
            test_acc
        );
    }
}

