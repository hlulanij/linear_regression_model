use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct LinearRegressionModel<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LinearRegressionModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(1, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}
