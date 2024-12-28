
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use mnist::MnistBuilder;

/// Represents a single sample (input + target).
#[derive(Clone, Debug)]
struct Sample {
    inputs: Vec<f32>,  // Normalized input pixels
    target: Vec<f32>,  // One-hot encoded label
}

/// Represents a single neuron in the network.
#[derive(Debug)]
pub struct Neuron {
    raw_value: f32,       // Weighted sum (z)
    weights: Vec<f32>,    // Incoming weights
    bias: f32,            // Bias
    delta: f32,           // Delta for backprop
    activated_value: f32, // Activated output (a)
}

impl Neuron {
    /// Creates a new Neuron with random weights and biases.
    pub fn new(num_inputs: usize, activation: &str) -> Self {
        let mut rng = thread_rng();
        // Heuristic scale for different activations. We'll just use "sigmoid" or default here.
        let scale = match activation {
            "sigmoid" => (1.0 / num_inputs as f32).sqrt(),
            _ => 0.5,
        };

        let weights: Vec<f32> = (0..num_inputs)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        Neuron {
            raw_value: 0.0,
            weights,
            bias: rng.gen_range(-scale..scale),
            delta: 0.0,
            activated_value: 0.0,
        }
    }
}

/// Represents a layer in the network.
#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a new Layer with `num_neurons` neurons, each with `num_inputs` incoming weights.
    pub fn new(num_neurons: usize, num_inputs: usize, activation: &str) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..num_neurons {
            neurons.push(Neuron::new(num_inputs, activation));
        }
        Layer { neurons }
    }
}

/// Activation functions
mod activations {
    /// Sigmoid activation
    pub fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Derivative of sigmoid with respect to `z`
    pub fn sigmoid_derivative(z: f32) -> f32 {
        let s = sigmoid(z);
        s * (1.0 - s)
    }

    /// Softmax for output layer
    pub fn softmax(z: &[f32]) -> Vec<f32> {
        let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = z.iter().map(|&x| (x - max_z).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum_exps).collect()
    }
}

/// Loads, normalizes, and prepares the MNIST dataset.
fn load_mnist() -> (Vec<Sample>, Vec<Sample>) {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();

    // Convert training images
    let train_images = normalize_images(mnist.trn_img);
    let train_labels = mnist.trn_lbl;
    let train_set = convert_to_samples(&train_images, &train_labels);

    // Convert test images
    let test_images = normalize_images(mnist.tst_img);
    let test_labels = mnist.tst_lbl;
    let test_set = convert_to_samples(&test_images, &test_labels);

    (train_set, test_set)
}

/// Normalizes MNIST image pixels to [0,1].
fn normalize_images(images: Vec<u8>) -> Vec<f32> {
    images.into_iter().map(|p| p as f32 / 255.0).collect()
}

/// One-hot encodes digit labels into 10-element vectors.
fn one_hot_encode(label: u8, num_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0; num_classes];
    if (label as usize) < num_classes {
        v[label as usize] = 1.0;
    }
    v
}

/// Converts raw images and labels into `Sample` structs.
fn convert_to_samples(images: &[f32], labels: &[u8]) -> Vec<Sample> {
    images
        .chunks(784) // 28x28
        .zip(labels.iter())
        .map(|(img, &lab)| Sample {
            inputs: img.to_vec(),
            target: one_hot_encode(lab, 10),
        })
        .collect()
}

/// Builds the network layer by layer.
fn initialize_network(layer_sizes: &[usize]) -> Vec<Layer> {
    let mut layers = Vec::new();

    // Input layer: no incoming weights, so num_inputs=0
    layers.push(Layer::new(layer_sizes[0], 0, "none"));

    // Hidden + Output layers
    for i in 1..layer_sizes.len() {
        let activation = if i == layer_sizes.len() - 1 {
            "softmax"
        } else {
            "sigmoid"
        };
        layers.push(Layer::new(layer_sizes[i], layer_sizes[i - 1], activation));
    }

    layers
}

/// Forward pass: calculates activated values for each layer.

fn forward_pass(layers: &mut [Layer], inputs: &[f32]) {
    // Pre-capture the total number of layers to avoid calling layers.len() later.
    let total_layers = layers.len();

    // 1. Set the activations for the input layer.
    for (i, neuron) in layers[0].neurons.iter_mut().enumerate() {
        neuron.raw_value = inputs[i];
        neuron.activated_value = inputs[i];
    }

    // 2. For each subsequent layer, gather the previous layer's activations first.
    for l in 1..total_layers {
        // Gather previous layer's activations into a local vector (no overlapping borrow).
        let prev_activations: Vec<f32> = layers[l - 1]
            .neurons
            .iter()
            .map(|n| n.activated_value)
            .collect();

        // Check if this is not the final (output) layer.
        // We store the result outside the loop to avoid borrowing `layers` again.
        let is_not_output = l < (total_layers - 1);

        // Mutate the current layer (no conflicting borrow).
        for neuron in &mut layers[l].neurons {
            let mut weighted_sum = 0.0;

            // Accumulate weights * previous activations.
            for (k, &prev_val) in prev_activations.iter().enumerate() {
                weighted_sum += neuron.weights[k] * prev_val;
            }
            weighted_sum += neuron.bias;

            // Store the raw_value.
            neuron.raw_value = weighted_sum;

            // If not output layer => apply sigmoid, otherwise just keep raw_value for softmax.
            if is_not_output {
                neuron.activated_value = activations::sigmoid(weighted_sum);
            } else {
                neuron.activated_value = weighted_sum;
            }
        }
    }

    // 3. Apply softmax to the final (output) layer.
    //    (We do not borrow `layers` mutably here, just read its length from total_layers).
    let output_idx = total_layers - 1;
    let raw_outputs: Vec<f32> = layers[output_idx]
        .neurons
        .iter()
        .map(|n| n.raw_value)
        .collect();

    let softmax_vals = activations::softmax(&raw_outputs);

    for (neuron, &val) in layers[output_idx].neurons.iter_mut().zip(softmax_vals.iter()) {
        neuron.activated_value = val;
    }
}
/// Backpropagates errors and updates weights/biases.
fn back_propagate(layers: &mut [Layer], targets: &[f32], lr: f32) {
    let out_idx = layers.len() - 1;

    // 1. Output layer deltas
    for (i, neuron) in layers[out_idx].neurons.iter_mut().enumerate() {
        neuron.delta = neuron.activated_value - targets[i];
    }

    println!("Output Layer Deltas:");
    for (i, neuron) in layers[out_idx].neurons.iter().enumerate() {
        println!("  Neuron {}: delta = {:.6}", i, neuron.delta);
    }

    // 2. Hidden layers deltas (in reverse)
    for l in (1..out_idx).rev() {
        // gather next layer's deltas + weights
        let next_deltas: Vec<f32> = layers[l + 1].neurons.iter().map(|n| n.delta).collect();
        let next_weights: Vec<Vec<f32>> = layers[l + 1]
            .neurons
            .iter()
            .map(|n| n.weights.clone())
            .collect();

        for (j, neuron) in layers[l].neurons.iter_mut().enumerate() {
            let mut sum = 0.0;
            // Summation of next_deltas[m] * next_weights[m][j]
            for (m, &delta_val) in next_deltas.iter().enumerate() {
                sum += delta_val * next_weights[m][j];
            }
            neuron.delta = sum * activations::sigmoid_derivative(neuron.raw_value);
        }
    }

    println!("Hidden Layer Deltas:");
    for l in 1..out_idx {
        // Print deltas for first 5 neurons in hidden layer
        for (j, neuron) in layers[l].neurons.iter().enumerate().take(5) {
            println!("  Layer {}, Neuron {}: delta = {:.6}", l, j, neuron.delta);
        }
    }

    // 3. Update weights + biases for each layer except input
    for l in 1..layers.len() {
        // gather previous layer's activations
        let prev_activations: Vec<f32> = layers[l - 1]
            .neurons
            .iter()
            .map(|n| n.activated_value)
            .collect();

        // mutate the current layer
        for neuron in &mut layers[l].neurons {
            // bias update
            neuron.bias -= lr * neuron.delta;
            // weight updates
            for (k, &prev_val) in prev_activations.iter().enumerate() {
                let gradient = neuron.delta * prev_val;
                neuron.weights[k] -= lr * gradient;
            }
        }
    }
}

/// Computes cross-entropy loss for the final layer.
fn calculate_loss(layers: &[Layer], targets: &[f32]) -> f32 {
    let out_idx = layers.len() - 1;
    layers[out_idx]
        .neurons
        .iter()
        .zip(targets.iter())
        .map(|(n, &t)| -t * (n.activated_value + 1e-12).ln())
        .sum()
}

/// Evaluates accuracy by comparing predictions vs. actual labels.
fn evaluate(layers: &mut [Layer], dataset: &[Sample]) -> f32 {
    let out_idx = layers.len() - 1;
    let mut correct = 0;
    for sample in dataset {
        forward_pass(layers, &sample.inputs);
        // find prediction
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

/// Returns index of maximum value in `vals`.
fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Randomly shuffles the dataset in-place.
fn shuffle_dataset(dataset: &mut [Sample]) {
    let mut rng = thread_rng();
    dataset.shuffle(&mut rng);
}

/// Training loop: shuffles data, runs forward + backprop, logs progress.
fn train(
    layers: &mut [Layer],
    training_set: &mut [Sample],
    epochs: usize,
    learning_rate: f32,
    test_set: &mut [Sample],
) {
    for epoch in 0..epochs {
        shuffle_dataset(training_set);

        let mut total_loss = 0.0;
        for sample in training_set.iter() {
            forward_pass(layers, &sample.inputs);
            back_propagate(layers, &sample.target, learning_rate);
            total_loss += calculate_loss(layers, &sample.target);
        }

        // Evaluate on the training set or test set
        let train_acc = evaluate(layers, training_set);
        let test_acc = evaluate(layers, test_set);

        println!("Epoch: {}", epoch);
        println!(
            "Training Loss: {:.4}, Training Acc: {:.2}%, Test Acc: {:.2}%",
            total_loss / training_set.len() as f32,
            train_acc,
            test_acc
        );
        println!();
    }
}

fn main() {
    // 1. Load dataset
    let (train_set, mut test_set) = load_mnist();

    println!("Loaded training samples: {}", train_set.len());
    println!("Loaded testing samples: {}", test_set.len());

    // Print sample data
    if let Some(sample) = train_set.get(0) {
        println!("First Training Sample, first 10 inputs: {:?}", &sample.inputs[..10]);
        println!("Target: {:?}", sample.target);
    }

    // 2. Filter + select subset for training
    // (ensures we get some non-zero input data)
    let num_training_samples = 1000;
    let mut filtered_train: Vec<Sample> = train_set
        .into_iter()
        .filter(|s| s.inputs.iter().any(|&v| v > 0.0))
        .take(num_training_samples)
        .collect();

    println!("Filtered training set to {} samples with non-zero inputs.", filtered_train.len());

    // 3. Build network
    let layer_sizes = vec![784, 128, 10]; // [input, hidden, output]
    let mut layers = initialize_network(&layer_sizes);

    // 4. Inspect a few weights/biases before training
    println!("\nInitial hidden layer neuron 0 weights[0], bias:");
    println!(
        "Weight[0]: {:.6}, Bias: {:.6}",
        layers[1].neurons[0].weights[0], layers[1].neurons[0].bias
    );

    // 5. Train
    let learning_rate = 0.1;
    let epochs = 10;
    train(&mut layers, &mut filtered_train, epochs, learning_rate, &mut test_set);

    // 6. Inspect same neuron after training
    println!("\nUpdated hidden layer neuron 0 weights[0], bias:");
    println!(
        "Weight[0]: {:.6}, Bias: {:.6}",
        layers[1].neurons[0].weights[0], layers[1].neurons[0].bias
    );

    println!("Done training.");
}

