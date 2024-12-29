
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

pub fn sigmoid_derivative(z: f32) -> f32 {
    let s = sigmoid(z);
    s * (1.0 - s)
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|&x| (x - max_z).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum_exps).collect();
}
