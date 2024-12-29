use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn shuffle_dataset<T>(dataset: &mut [T]) {
    let mut rng = thread_rng();
    dataset.shuffle(&mut rng);
}


