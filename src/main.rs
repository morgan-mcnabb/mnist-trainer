
use eframe::NativeOptions;
use neural_net::gui::GuiApp;
use std::panic;

fn main() {
    let native_options = NativeOptions::default();
    panic::set_hook(Box::new(|info| {
        // Print or log the panic info
        println!("Panic occurred: {:?}", info);
    }));
    let _ =eframe::run_native(
        "Neural Network GUI",
        native_options,
        Box::new(|_cc| Ok(Box::new(GuiApp::default()))),
    );
}

/*
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let (train_set, test_set) = load_mnist();

    info!("Loaded training samples: {}", train_set.len());
    info!("Loaded testing samples: {}", test_set.len());

    /*let num_training_samples = 10000;
    let mut shuffled_train_set = train_set.clone();
    let mut rng = thread_rng();
    shuffled_train_set.shuffle(&mut rng);
    let filtered_train = train_set
        .into_iter()
        .filter(|s| s.inputs.iter().any(|&v| v > 0.0)) 
        .take(num_training_samples)
        .collect::<Vec<_>>();

    info!(
        "Filtered training set to {} samples with non-zero inputs.",
        filtered_train.len()
    );
*/
    let layer_sizes = vec![784, 256, 128, 64, 10]; // [input, hidden1, hidden2, output]
    let activations = vec![Activation::Sigmoid, Activation::Sigmoid, Activation::Sigmoid]; // for hidden layers

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
    let epochs = 100;
    train(&mut network, &train_set, epochs, learning_rate, &test_set);

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
}*/
