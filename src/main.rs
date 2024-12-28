use rand::Rng;

fn main() {
    let _and_training_set = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ];

    let _or_training_set = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];

    let nand_training_set = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];

    let xor_training_set = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];

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
    }
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
            layers[l].neurons[j].raw_value = weighted_sum;
            layers[l].neurons[j].activated_value =
                sigmoid(layers[l].neurons[j].raw_value);
        }
    }
}

pub fn back_propagate(layers: &mut Vec<Layer>, targets: Vec<f32>, learning_rate: f32) {
    // compute deltas for output layer
    let output_layer_index = layers.len() - 1;
    for i in 0..layers[output_layer_index].neurons.len() {
        let output_val = layers[output_layer_index].neurons[i].activated_value;
        let target_val = targets[i];
        let error_signal = output_val - target_val;
        layers[output_layer_index].neurons[i].delta = error_signal
            * sigmoid_derivative(layers[output_layer_index].neurons[i].activated_value);
    }

    // compute deltas for hidden layer(s)
    for i in (1..output_layer_index).rev() {
        for j in 0..layers[i].neurons.len() {
            let mut sum_deltas = 0.0;
            for (m, next_neuron) in layers[i + 1].neurons.iter().enumerate() {
                sum_deltas += next_neuron.delta * layers[i + 1].neurons[m].weights[j];
            }
            let hidden_activated_val = layers[i].neurons[j].activated_value;
            layers[i].neurons[j].delta =
                sum_deltas * sigmoid_derivative(hidden_activated_val);
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

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-value))
}

fn sigmoid_derivative(value: f32) -> f32 {
    value * (1.0 - value)
}

fn relu(z: f32) -> f32 {
    if z > 0.0 {
        return z; 
    }

    0.0
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
