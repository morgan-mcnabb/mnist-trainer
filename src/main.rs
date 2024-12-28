use rand::Rng;
use mnist::{Mnist, MnistBuilder};

fn main() {

    let mnist = load_mnist();
    let (train_set, test_set) = prepare_dataset(mnist);

                        // input, hidden, output                            
    let layers_neurons = vec![784, 128, 10];
    let mut layers = initialize_network(&layers_neurons);

    let learning_rate = 0.001;
    let epochs = 10_000;


    train(&mut layers, &train_set, epochs, learning_rate, &test_set);

/*
    let mut rng = rand::thread_rng();
    let mut layers: Vec<Layer> = vec![];
    layers.push(Layer::new(2, 0));
    layers.push(Layer::new(5, layers[0].neurons.len()));
    layers.push(Layer::new(1, layers[1].neurons.len()));

    let learning_rate = 0.1;

    for epoch in 0..40001 {
        for values in xor_training_set {
            let mut inputs: Vec<f32> = vec![];
            let mut target: Vec<f32> = vec![];
            inputs.push(values[0]);
            inputs.push(values[1]);
            target.push(values[2]);

            forward_pass(&mut layers, inputs);
            back_propagate(&mut layers, target, learning_rate);
        }
        if epoch % 1000 == 0 {
            println!("Epoch: {}", epoch);
            let mut total_loss = 0.0;

            for values in xor_training_set {
                let inputs = vec![values[0], values[1]];
                let target = vec![values[2]];
                forward_pass(&mut layers, inputs);
                let output_layer_index = layers.len() - 1;
                let output_val = layers[output_layer_index].neurons[0].activated_value;  
                total_loss += calculate_loss(&layers, &target);
                println!(
                    "Input: ({}, {}), Expected: {}, Got: {:.4}",
                    values[0], values[1], values[2], output_val
                );

                println!("Loss: {:.4}", total_loss / xor_training_set.len() as f32);
            }
            println!();
        }
    }*/
}

pub fn forward_pass(layers: &mut Vec<Layer>, inputs: Vec<f32>) {
    // set inputs
    for i in 0..layers[0].neurons.len() {
        layers[0].neurons[i].raw_value = inputs[i];
        layers[0].neurons[i].activated_value = inputs[i];
    }

    for l in 1..layers.len() {
        for j in 0..layers[l].neurons.len() {
            let mut weighted_sum = 0.0;
            for k in 0..layers[l - 1].neurons.len() {
                weighted_sum +=
                    layers[l].neurons[j].weights[k] * layers[l - 1].neurons[k].activated_value;
            }
            weighted_sum += layers[l].neurons[j].bias;
            if l < layers.len() - 1 {
                layers[l].neurons[j].activated_value = relu(weighted_sum);
            }
            else {
                layers[l].neurons[j].activated_value = weighted_sum;
            }

            //layers[l].neurons[j].raw_value = weighted_sum;
            //layers[l].neurons[j].activated_value =
            //    sigmoid(layers[l].neurons[j].raw_value);
        }
    }

    let output_layer_ndx = layers.len() - 1;
    let output_z = layers[output_layer_ndx]
        .neurons
        .iter()
        .map(|neuron| neuron.raw_value)
        .collect::<Vec<f32>>();

    let output_softmax = softmax(&output_z);
    for i in 0..layers[output_layer_ndx].neurons.len() {
        layers[output_layer_ndx].neurons[i].activated_value = output_softmax[i];
    }
}

pub fn back_propagate(layers: &mut Vec<Layer>, targets: Vec<f32>, learning_rate: f32) {
    // compute deltas for output layer
    let output_layer_index = layers.len() - 1;


    for i in 0..layers[output_layer_index].neurons.len() {
        let output_val = layers[output_layer_index].neurons[i].activated_value;
        let target_val = targets[i];
        //let error_signal = output_val - target_val;
        //layers[output_layer_index].neurons[i].delta = error_signal
        //    * sigmoid_derivative(layers[output_layer_index].neurons[i].activated_value);
        layers[output_layer_index].neurons[i].delta = output_val - target_val;
    }

    // compute deltas for hidden layer(s)
    for i in (1..output_layer_index).rev() {
        for j in 0..layers[i].neurons.len() {
            let mut sum_deltas = 0.0;
            //for (m, next_neuron) in layers[i + 1].neurons.iter().enumerate() {
            //    sum_deltas += next_neuron.delta * layers[i + 1].neurons[m].weights[j];
            //}
            for m in 0..layers[i + 1].neurons.len() {
                sum_deltas += layers[i + 1].neurons[m].delta * layers[i].neurons[j].weights[m];
            }
            
            //let hidden_activated_val = layers[i].neurons[j].activated_value;
            //layers[i].neurons[j].delta =
            //sum_deltas * sigmoid_derivative(hidden_activated_val);
            let hidden_raw = layers[i].neurons[j].raw_value;
            layers[i].neurons[j].delta = sum_deltas * relu_derivative(hidden_raw);
        }
    }

    // update weights and biases
    for l in 1..layers.len() {
        for j in 0..layers[l].neurons.len() {
            let delta = layers[l].neurons[j].delta;
            layers[l].neurons[j].bias -= learning_rate * delta;

            // update weights
            for k in 0..layers[l - 1].neurons.len() {
                let prev_activated = layers[l - 1].neurons[k].activated_value;
                let gradient = delta * prev_activated;
                layers[l].neurons[j].weights[k] -= learning_rate * gradient;
            }
        }
    }
}

#[derive(Debug)]
pub struct Neuron {
    raw_value: f32,
    weights: Vec<f32>,
    bias: f32,
    delta: f32,
    activated_value: f32,
}

impl Neuron {
    pub fn new(initial_val: f32, num_weights: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (6.0 / (num_weights as f32 + 1.0)).sqrt();
        let weights: Vec<f32> = (0..num_weights).map(|_| rng.gen_range(-scale..scale)).collect();
        Neuron {
            raw_value: initial_val,
            weights: weights,
            bias: rng.gen::<f32>(),
            delta: 0.0,
            activated_value: 0.0,
        }
    }

    pub fn set_raw_value(&mut self, new_val: f32) {
        self.raw_value = new_val;
    }

    pub fn set_delta(&mut self, val: f32) {
        self.delta = val;
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Layer {
            neurons: (0..num_neurons)
                .map(|_| Neuron {
                    raw_value: 0.0,
                    activated_value: 0.0,
                    weights: (0..num_inputs).map(|_| rng.gen_range(-0.5..0.5)).collect(),
                    bias: rng.gen_range(-0.5..0.5),
                    delta: 0.0,
                })
                .collect(),
        }
    }

    fn set_neuron_raw_values(&mut self, inputs: &[f32]) {
        // could be problems if inputs is a differnet length than neurons.len
        for n in 0..self.neurons.len() {
            self.neurons[n].raw_value = inputs[n];
        }
    }
}

fn initialize_network(layers_num_neurons: &Vec<usize>) -> Vec<Layer> {
    let mut layers = Vec::new();

    layers.push(Layer::new(layers_num_neurons[0], 0));

    for layer in 1..layers_num_neurons.len() {
        layers.push(Layer::new(layers_num_neurons[layer], layers_num_neurons[layer - 1]));
    }

    layers
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-value))
}

fn sigmoid_derivative(value: f32) -> f32 {
    value * (1.0 - value)
}

fn relu(z: f32) -> f32 {
   z.max(0.0) 
}

fn relu_derivative(z: f32) -> f32 {
    if z > 0.0 {
        return 1.0;
    }

    0.0
}

fn calculate_loss(layers: &Vec<Layer>, targets: &Vec<f32>) -> f32 {
    let output_layer = &layers[layers.len() - 1];
    let mut loss = 0.0;
    for i in 0..output_layer.neurons.len() {
        let output = output_layer.neurons[i].activated_value;
        let target = targets[i];
        loss += (output - target).powi(2);
    }
    loss / (output_layer.neurons.len() as f32)
}

fn softmax(z: &Vec<f32>) -> Vec<f32> {
    let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|&x| (x - max_z).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum_exps).collect()
}

fn cross_entropy_loss(predicted: &Vec<f32>, target: &Vec<f32>) -> f32 {
    let epsilon = 1e-12;
    predicted.iter().zip(target.iter()).map(|(&p, &t)| {
        -t * (p + epsilon).ln()
    }).sum::<f32>()
}

fn load_mnist() -> Mnist {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();
    mnist
}

fn normalize_images(images: Vec<u8>) -> Vec<f32> {
    images.into_iter().map(|pixel| pixel as f32 / 255.0).collect()
}

fn one_hot_encode(label: u8, num_classes: usize) -> Vec<f32> {
    let mut encoded = vec![0.0; num_classes];
    if(label as usize) < num_classes {
        encoded[label as usize] = 1.0;
    }
    
    encoded
}

struct Sample {
    inputs: Vec<f32>,
    target: Vec<f32>,
}

fn prepare_dataset(mnist: Mnist) -> (Vec<Sample>, Vec<Sample>) {
    let train_images = normalize_images(mnist.trn_img);
    let train_labels = mnist.trn_lbl;

    let test_images = normalize_images(mnist.tst_img);
    let test_labels = mnist.tst_lbl;

    let train_set = convert_to_samples(&train_images, &train_labels);

    let test_set = convert_to_samples(&test_images, &test_labels);

    (train_set, test_set)
}

fn convert_to_samples(images: &Vec<f32>, labels: &Vec<u8>) -> Vec<Sample> {
    images
        .chunks(784) // 28 x 28 image = 784 pixels
        .zip(labels.iter())
        .map(|(img, &label)| Sample {
            inputs: img.to_vec(),
            target: one_hot_encode(label, 10),
        })
        .collect::<Vec<Sample>>()

}

fn train(network: &mut Vec<Layer>, training_set: &Vec<Sample>, epochs: usize, learning_rate: f32, test_set: &Vec<Sample>) {
    for epoch in 0..epochs {
        for sample in training_set {
            forward_pass(network, sample.inputs.clone());
            back_propagate(network, sample.target.clone(), learning_rate);
        }

        if epoch % 1000 == 0 {
            println!("Epoch: {}", epoch);

            let training_accuracy = evaluate(network, training_set);
            let test_accuracy = evaluate(network, test_set);
            println!("Training Accuracy: {:.2}%, Test Accuract: {:.2}%", training_accuracy, test_accuracy);
            println!();
        }
    }
}

fn evaluate(network: &mut Vec<Layer>, dataset: &Vec<Sample>) -> f32 {
    let output_layer_ndx = network.len() - 1;
    let mut correct = 0;

    for sample in dataset {
        forward_pass(network, sample.inputs.clone());
        let outputs = &network[output_layer_ndx].neurons;
        let predicted = argmax(
            &outputs
                .iter()
                .map(|neuron| neuron.activated_value)
                .collect::<Vec<f32>>(),
        );

        let actual = argmax(&sample.target);
        if predicted == actual {
            correct += 1;
        }
    } 

    (correct as f32 / dataset.len() as f32) * 100.0
}

fn argmax(vec: &Vec<f32>) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
