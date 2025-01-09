use crate::network::layer::Layer;
use crate::data::dataset::Sample;
use crate::utils::math::shuffle_dataset;
use ndarray::Array1;
use crate::network::activation::{Activation,softmax};

#[derive(Clone)]
struct NeuronGradients {
    d_bias: f32,
    d_weights: Array1<f32>,
}

#[derive(Clone)] 
struct LayerGradients {
    neuron_gradients: Vec<NeuronGradients>,
}

pub fn forward_pass(layers: &mut [Layer], inputs: &Array1<f32>) {
    let total_layers = layers.len();

    for (i, neuron) in layers[0].neurons.iter_mut().enumerate() {
        neuron.raw_value = inputs[i];
        neuron.activated_value = inputs[i];
    }

    for l in 1..total_layers {
        let prev_activations = layers[l - 1].activated_values();

        let is_not_output = l < (total_layers - 1);

        for neuron in &mut layers[l].neurons {
            let weighted_sum = neuron.weights.dot(&prev_activations) + neuron.bias;
            neuron.raw_value = weighted_sum;

            if let Some(activation) = layers[l].activation {
                match activation {
                    Activation::Softmax => {
                        neuron.activated_value = weighted_sum;
                    }
                    _ => {
                        neuron.activated_value = activation.activate(weighted_sum)
                    }
                }
            }


            neuron.activated_value = if is_not_output {
                let activation = layers[l].activation.unwrap();
                activation.activate(weighted_sum)
            } else {
                weighted_sum // softmax will be applied later
            };
        }
    }

    // apply softmax now
    let output_index = total_layers - 1;
    let raw_outputs: Array1<f32> = layers[output_index]
        .neurons
        .iter()
        .map(|n| n.raw_value)
        .collect();
    let softmax_values = softmax(&raw_outputs);

    for (neuron, &val) in layers[output_index].neurons.iter_mut().zip(softmax_values.iter()) {
        neuron.activated_value = val;
    }
}

fn initialize_layer_gradients(layers: &[Layer]) -> Vec<LayerGradients> {
    let mut gradients = Vec::with_capacity(layers.len());
    for layer in layers {
        let mut neuron_gradients = Vec::with_capacity(layer.neurons.len());
        for neuron in &layer.neurons {
            neuron_gradients.push(NeuronGradients {
                d_bias: 0.0,
                d_weights: Array1::zeros(neuron.weights.len()),
            });
        }
        gradients.push(LayerGradients { neuron_gradients });
    }

    gradients
}

fn accumulate_gradients(
    layers: &[Layer],
    sample_grads: &[Layer],
    batch_grads: &mut [LayerGradients],
) {
    // sample_grads is a "layer-like" structure storing .delta in each neuron
    // along with the original weights/bias. We need only the deltas, though.
    // We'll add them to batch_grads.
    for (layer_idx, layer) in layers.iter().enumerate().skip(1) {
        let prev_activations = layers[layer_idx - 1].activated_values();

        for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
            // The gradient w.r.t. bias = delta
            let delta = neuron.delta;

            // Accumulate bias gradient
            batch_grads[layer_idx].neuron_gradients[neuron_idx].d_bias += delta;

            // Accumulate weights gradient
            // gradient for each weight_i = prev_activation_i * delta
            for (w_idx, &prev_a) in prev_activations.iter().enumerate() {
                batch_grads[layer_idx].neuron_gradients[neuron_idx].d_weights[w_idx] += prev_a * delta;
            }
        }
    }
}

/// Apply the accumulated gradients in `batch_grads` to the actual network
/// weights/biases, then reset `batch_grads` to zero for next batch.
fn apply_gradients(layers: &mut [Layer], batch_grads: &mut [LayerGradients], batch_size: f32, lr: f32) {
    for (layer_idx, layer) in layers.iter_mut().enumerate().skip(1) {
        for (neuron_idx, neuron) in layer.neurons.iter_mut().enumerate() {
            let grad = &mut batch_grads[layer_idx].neuron_gradients[neuron_idx];

            // Average gradient = sum / batch_size
            let avg_db = grad.d_bias / batch_size;
            neuron.bias -= lr * avg_db;

            for w_idx in 0..neuron.weights.len() {
                let avg_dw = grad.d_weights[w_idx] / batch_size;
                neuron.weights[w_idx] -= lr * avg_dw;
            }

            // Reset for next batch
            grad.d_bias = 0.0;
            grad.d_weights.fill(0.0);
        }
    }
}

pub fn back_propagate_batch(layers: &mut [Layer], targets: &Array1<f32>, learning_rate: f32) {
    let output_index = layers.len() - 1;

    for (i, neuron) in layers[output_index].neurons.iter_mut().enumerate() {
        neuron.delta = neuron.activated_value - targets[i];
    }

    for l in (1..output_index).rev() {
        let next_layer_deltas: Vec<f32> = layers[l + 1].neurons.iter().map(|n| n.delta).collect();
        let next_layer_weights: Vec<Vec<f32>> = layers[l + 1].neurons.iter().map(|n| n.weights.to_vec()).collect();
        let activation = layers[l].activation.unwrap();

        for (j, neuron) in layers[l].neurons.iter_mut().enumerate() {
            let sum: f32 = next_layer_deltas.iter().zip(next_layer_weights.iter()).map(|(delta, weights)| delta * weights[j]).sum();
            neuron.delta = sum * activation.derivate(neuron.activated_value);
        }
    }

    // commenting this out so we can use batch processing!!
    /*
    for l in 1..layers.len() {
        let prev_activations = layers[l - 1].activated_values();

        for neuron in &mut layers[l].neurons {
            neuron.bias -= learning_rate * neuron.delta;

            let gradient = &prev_activations * neuron.delta;
            neuron.weights = &neuron.weights - &(gradient * learning_rate);
        }
    }*/
}

pub fn back_propagate(layers: &mut [Layer], targets: &Array1<f32>, learning_rate: f32) {
    let output_index = layers.len() - 1;

    for (i, neuron) in layers[output_index].neurons.iter_mut().enumerate() {
        neuron.delta = neuron.activated_value - targets[i];
    }

    for l in (1..output_index).rev() {
        let next_layer_deltas: Vec<f32> = layers[l + 1].neurons.iter().map(|n| n.delta).collect();
        let next_layer_weights: Vec<Vec<f32>> = layers[l + 1].neurons.iter().map(|n| n.weights.to_vec()).collect();
        let activation = layers[l].activation.unwrap();

        for (j, neuron) in layers[l].neurons.iter_mut().enumerate() {
            let sum: f32 = next_layer_deltas.iter().zip(next_layer_weights.iter()).map(|(delta, weights)| delta * weights[j]).sum();
            neuron.delta = sum * activation.derivate(neuron.activated_value);
        }
    }

    // commenting this out so we can use batch processing!!
    
    for l in 1..layers.len() {
        let prev_activations = layers[l - 1].activated_values();

        for neuron in &mut layers[l].neurons {
            neuron.bias -= learning_rate * neuron.delta;

            let gradient = &prev_activations * neuron.delta;
            neuron.weights = &neuron.weights - &(gradient * learning_rate);
        }
    }
}


/*fn calculate_loss(layers: &[Layer], targets: &Array1<f32>) -> f32 {
    let output_index = layers.len() - 1;
    layers[output_index]
        .neurons
        .iter()
        .zip(targets.iter())
        .map(|(neuron, &target)| -target * (neuron.activated_value + 1e-12).ln())
        .sum()
}*/

pub fn train_with_minibatch(
    layers: &mut [Layer],
    training_set: &[Sample],
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
) {
    for _ in 0..epochs {
        // Shuffle the training data each epoch
        let mut shuffled = training_set.to_vec();
        shuffle_dataset(&mut shuffled);

        // Process data in batches
        for batch in shuffled.chunks(batch_size) {
            // 1) Initialize zeroed-out gradients for this batch
            let mut batch_grads = initialize_layer_gradients(layers);

            // 2) For each sample in the batch, do forward pass and backprop,
            //    accumulate partial derivatives.
            for sample in batch {
                forward_pass(layers, &sample.inputs);
                back_propagate_batch(layers, &sample.target, 0.0);
                // Now add these "delta" values to the batch gradients
                accumulate_gradients(layers, layers, &mut batch_grads);
            }

            // 3) After the entire batch, update the weights/biases once
            apply_gradients(layers, &mut batch_grads, batch.len() as f32, learning_rate);
        }
    }
}

pub fn train(
    layers: &mut [Layer],
    training_set: &[Sample],
    epochs: usize,
    learning_rate: f32,
    //test_set: &[Sample],
) {
    for _ in 0..epochs {
        let mut shuffled = training_set.to_vec();
        shuffle_dataset(&mut shuffled);

        for sample in shuffled.iter() {
            forward_pass(layers, &sample.inputs);
            back_propagate(layers, &sample.target, learning_rate);
        }

        }
}
