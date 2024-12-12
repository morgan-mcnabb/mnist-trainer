use rand::Rng;

fn main() {
    let and_training_set = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ];

    let or_training_set = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ];

    let nand_training_set = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ];

    let mut rng = rand::thread_rng();

    // Initialize weights and biases
    let learning_rate = 0.1;
    let mut input_weight_1 = rng.gen_range(-0.5..0.5);
    let mut input_weight_2 = rng.gen_range(-0.5..0.5);
    let mut hidden_neuron_weight = rng.gen_range(-0.5..0.5);
    let mut hidden_bias = rng.gen_range(-0.5..0.5);
    let mut output_bias = rng.gen_range(-0.5..0.5);

    for epoch in 0..40000 {
        for values in nand_training_set {
            let x1 = values[0];
            let x2 = values[1];
            let target = values[2];

            // Forward pass
            let hidden_input = x1 * input_weight_1 + x2 * input_weight_2 + hidden_bias;
            let hidden_neuron = activation_function(hidden_input);

            let output_input = hidden_neuron * hidden_neuron_weight + output_bias;
            let output_neuron = activation_function(output_input);

            // Calculate error and output delta
            let output_diff = target - output_neuron;
            let delta_output = output_diff * activation_function_derivative(output_neuron);// output_neuron * (1.0 - output_neuron);

            // Calculate hidden delta
            let delta_hidden = delta_output * hidden_neuron_weight * activation_function_derivative(hidden_neuron);//hidden_neuron * (1.0 - hidden_neuron);
                                                                                                                  
            // Update weights and biases for output layer
            hidden_neuron_weight += learning_rate * delta_output * hidden_neuron;
            output_bias += learning_rate * delta_output;

            // Update weights and biases for hidden layer
            input_weight_1 += learning_rate * delta_hidden * x1;
            input_weight_2 += learning_rate * delta_hidden * x2;
            hidden_bias += learning_rate * delta_hidden;
        }

        // Optional: Print error every so often to monitor learning
        if epoch % 1000 == 0 {
            println!("Epoch: {}", epoch);
            for values in nand_training_set {
                let x1 = values[0];
                let x2 = values[1];
                let target = values[2];

                let hidden_input = x1 * input_weight_1 + x2 * input_weight_2 + hidden_bias;
                let hidden_neuron = activation_function(hidden_input);

                let output_input = hidden_neuron * hidden_neuron_weight + output_bias;
                let output_neuron = activation_function(output_input);

                println!(
                    "Input: ({}, {}) Expected: {} Got: {:.4}",
                    x1, x2, target, output_neuron
                );
            }
            println!();
        }
    }
}

pub struct Neuron {
    value: f32,
    weights: Vec<f32>,
}

fn activation_function(value: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-value))
}

fn activation_function_derivative(value: f32) -> f32 {
    value * (1.0 - value)
}


