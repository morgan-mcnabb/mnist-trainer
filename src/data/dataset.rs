
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct Sample {
    pub inputs: Array1<f32>,  
    pub target: Array1<f32>,  
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

fn normalize_images(image: &[u8]) -> Array1<f32> {
    Array1::from_iter(image.iter().map(|&p| p as f32 / 255.0))
}

fn one_hot_encode(label: u8, num_classes: usize) -> Array1<f32> {
    let mut encoding = Array1::zeros(num_classes);
    if (label as usize) < num_classes {
        encoding[label as usize] = 1.0;
    }
    encoding
}


