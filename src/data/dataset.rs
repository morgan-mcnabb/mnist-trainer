
#[derive(Clone, Debug)]
pub struct Sample {
    pub inputs: Vec<f32>,  // Normalized input pixels
    pub target: Vec<f32>,  // One-hot encoded label
}

pub fn create_samples(images: &[u8], labels: &[u8], num_classes: usize) -> Vec<Sample> {
    images
        .chunks(784) // 28x28
        .zip(labels.iter())
        .map(|(img, &lab)| Sample {
            inputs: normalize_images(img),
            target: one_hot_encode(lab, num_classes),
        })
        .collect()
}

fn normalize_images(image: &[u8]) -> Vec<f32> {
    image.iter().map(|&p| p as f32 / 255.0).collect()
}

fn one_hot_encode(label: u8, num_classes: usize) -> Vec<f32> {
    let mut encoding = vec![0.0; num_classes];
    if (label as usize) < num_classes {
        encoding[label as usize] = 1.0;
    }
    encoding
}
