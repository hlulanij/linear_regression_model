use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use burn::tensor::Tensor;
use burn::optim::{Optimizer, SgdConfig};
use crate::model::LinearRegressionModel;

pub(crate) fn train_model(
 x: Tensor<NdArray<f32>, 2>,
 y: Tensor<NdArray<f32>, 2>,
 device: &burn_ndarray::NdArrayDevice,
) -> LinearRegressionModel<NdArray<f32>> {
 let learning_rate = 0.01;
 let epochs = 1000;

 let mut model = LinearRegressionModel::<NdArray<f32>>::new(device);

 // ✅ Correctly initialize SGD optimizer
 let optim_config = SgdConfig::new().with_lr(learning_rate); // ✅ Use `with_lr`
 let mut optimizer = optim_config.init(); // ✅ Use `.init()` instead of `Sgd::new()`

 for _ in 0..epochs {
  // Forward pass
  let predictions = model.forward(x.clone());

  // Compute Mean Squared Error manually
  let loss = (predictions - y.clone()).powf(2.0).mean();

  // Backpropagation
  let grads = loss.backward();
  optimizer.step(&mut model, grads); // ✅ Correct optimizer update
 }

 model
}
