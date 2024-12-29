pub fn cross_entropy_loss(predicted: &[f32], targets: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| -t * (p + 1e-12).ln())
        .sum()
}
