use neural_net::config::Config;
use neural_net::data::loader::load_mnist;
use neural_net::network::initialize_network;
use neural_net::training::trainer::{train, train_with_minibatch};
use neural_net::network::activation::Activation;
use std::time::Instant;
use std::panic;

fn main() {
    // Optional: Set up panic hook for better error messages
    panic::set_hook(Box::new(|info| {
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            eprintln!("Panic occurred: {}", s);
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            eprintln!("Panic occurred: {}", s);
        } else {
            eprintln!("Panic occurred: {:?}", info.payload());
        }

        if let Some(location) = info.location() {
            eprintln!(
                "Location: file '{}' at line {}, column {}",
                location.file(),
                location.line(),
                location.column()
            );
        }
    }));

    // 1. Load MNIST dataset
    println!("Loading MNIST dataset...");
    let (train_set, test_set) = load_mnist();
    println!("Dataset loaded: {} training samples, {} test samples.", train_set.len(), test_set.len());

    // 2. Define configuration
    let config = Config {
        epochs: 10,               // Reduced epochs for quicker benchmarking
        learning_rate: 0.1,
        layers: vec![784, 128, 64, 10],
        activations: vec![
            "sigmoid".to_string(),
            "sigmoid".to_string(),
            "softmax".to_string(),
        ],
        batch_size: 32,            // Define a batch size for mini-batch training
    };

    // 3. Initialize two identical networks
    println!("Initializing networks...");
    let mut network_stochastic = initialize_network(&config.layers, &convert_activations(&config.activations));
    let mut network_minibatch = initialize_network(&config.layers, &convert_activations(&config.activations));

    // 4. Benchmark Stochastic (Online) Training
    println!("Starting Stochastic (Online) Training...");
    let start_stochastic = Instant::now();
    train(
        &mut network_stochastic,
        &train_set,
        config.epochs,
        config.learning_rate,
    );
    let duration_stochastic = start_stochastic.elapsed();
    println!(
        "Stochastic Training completed in: {:.2?}",
        duration_stochastic
    );

    // Optionally, evaluate accuracy
    let accuracy_stochastic = neural_net::metrics::accuracy::evaluate(&mut network_stochastic, &test_set);
    println!("Stochastic Training Test Accuracy: {:.2}%", accuracy_stochastic);

    // 5. Benchmark Mini-Batch Training
    println!("\nStarting Mini-Batch Training...");
    let start_minibatch = Instant::now();
    train_with_minibatch(
        &mut network_minibatch,
        &train_set,
        config.epochs,
        config.learning_rate,
        config.batch_size,
    );
    let duration_minibatch = start_minibatch.elapsed();
    println!(
        "Mini-Batch Training completed in: {:.2?}",
        duration_minibatch
    );

    // Optionally, evaluate accuracy
    let accuracy_minibatch = neural_net::metrics::accuracy::evaluate(&mut network_minibatch, &test_set);
    println!("Mini-Batch Training Test Accuracy: {:.2}%", accuracy_minibatch);

    // 6. Summary
    println!("\n--- Benchmark Summary ---");
    println!(
        "Stochastic Training Time: {:.2?}, Test Accuracy: {:.2}%",
        duration_stochastic, accuracy_stochastic
    );
    println!(
        "Mini-Batch Training Time: {:.2?}, Test Accuracy: {:.2}%",
        duration_minibatch, accuracy_minibatch
    );
}

/// Helper function to convert activation strings to Activation enums
fn convert_activations(activations: &[String]) -> Vec<Activation> {
    activations.iter().map(|s| match s.to_lowercase().as_str() {
        "sigmoid" => Activation::Sigmoid,
        "relu" => Activation::ReLU,
        "softmax" => Activation::Softmax,
        _ => {
            println!("Unknown activation '{}', defaulting to Sigmoid.", s);
            Activation::Sigmoid
        }
    }).collect()
}

/*
fn main() {
    let native_options = NativeOptions::default();
    panic::set_hook(Box::new(|info| {
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            eprintln!("Panic occurred: {}", s);
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            eprintln!("Panic occurred: {}", s);
        } else {
            eprintln!("Panic occurred: {:?}", info.payload());
        }

        if let Some(location) = info.location() {
            eprintln!(
                "Location: file '{}' at line {}, column {}",
                location.file(),
                location.line(),
                location.column()
            );
        }

    }));


    let _ =eframe::run_native(
        "Neural Network GUI",
        native_options,
        Box::new(|_cc| Ok(Box::new(GuiApp::default()))),
    );
}*/
