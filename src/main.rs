
use neural_net::data::loader::load_mnist;
use neural_net::network::initialize_network;
use neural_net::network::layer::Layer;
use neural_net::data::dataset::Sample;
use neural_net::network::activation::Activation;
use neural_net::training::trainer::{train, forward_pass};
use log::{info, error};
use env_logger;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let (train_set, test_set) = load_mnist();

    info!("Loaded training samples: {}", train_set.len());
    info!("Loaded testing samples: {}", test_set.len());

    let num_training_samples = 1000;
    let filtered_train = train_set
        .into_iter()
        .filter(|s| s.inputs.iter().any(|&v| v > 0.0)) // Ensure non-zero inputs
        .take(num_training_samples)
        .collect::<Vec<_>>();

    info!(
        "Filtered training set to {} samples with non-zero inputs.",
        filtered_train.len()
    );

    let layer_sizes = vec![784, 128, 64, 10]; // [input, hidden1, hidden2, output]
    let activations = vec![Activation::Sigmoid, Activation::Sigmoid]; // For hidden layers

    let mut network = initialize_network(&layer_sizes, &activations);

    if let Some(neuron) = network[1].neurons.get(0) {
        info!(
            "Initial Hidden Layer Neuron 0 - Weight[0]: {:.6}, Bias: {:.6}",
            neuron.weights[0], neuron.bias
        );
    } else {
        error!("Hidden layer does not have any neurons.");
    }

    let learning_rate = 0.1;
    let epochs = 10;
    train(&mut network, &filtered_train, epochs, learning_rate, &test_set);

    if let Some(neuron) = network[1].neurons.get(0) {
        info!(
            "Updated Hidden Layer Neuron 0 - Weight[0]: {:.6}, Bias: {:.6}",
            neuron.weights[0], neuron.bias
        );
    } else {
        error!("Hidden layer does not have any neurons.");
    }

    if let Some(first_test_sample) = test_set.first() {
        let predicted_label = predict(&mut network, first_test_sample);
        let actual_label = first_test_sample
            .target
            .iter()
            .position(|&v| v == 1.0)
            .unwrap_or(0);
        info!(
            "Prediction for the first test sample: {}, Actual Label: {}",
            predicted_label, actual_label
        );
    } else {
        error!("Test set is empty.");
    }

    info!("Training complete.");
}

fn predict(layers: &mut [Layer], sample: &Sample) -> usize {
    forward_pass(layers, &sample.inputs);
    let output_index = layers.len() - 1;
    layers[output_index]
        .neurons
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.activated_value.partial_cmp(&b.1.activated_value).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
