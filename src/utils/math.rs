use rand::seq::SliceRandom;
use rand::thread_rng;

/// Shuffles the dataset in-place.
pub fn shuffle_dataset<T>(dataset: &mut [T]) {
    let mut rng = thread_rng();
    dataset.shuffle(&mut rng);
}

