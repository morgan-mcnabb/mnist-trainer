
use crate::core::Sample;
use crate::helpers::{load_mnist, /* etc. */};

/// Possibly define dataset structures or iterators to batch samples.
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn shuffle(&mut self) {
        // shuffle logic
    }

    // maybe define `get_batch(&self, size: usize) -> &[Sample]`
}
