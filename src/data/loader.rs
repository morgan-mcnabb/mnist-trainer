use rand::seq::SliceRandom;
use mnist::{Mnist, MnistBuilder};
use crate::data::dataset::Sample;

pub fn load_mnist() -> (Vec<Sample>, Vec<Sample>) {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();

    let train_set = dataset::create_samples(&mnist.trn_img, &mnist.trn_lbl, 10);
    let test_set = dataset::create_samples(&mnist.tst_img, &mnist.tst_lbl, 10);

    (train_set, test_set)
}
