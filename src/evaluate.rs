use burn::tensor::{Tensor, backend::Backend};
use crate::model::LinearRegressionModel;

pub fn evaluate_model<B: Backend>(
    model: &LinearRegressionModel<B>,
    x_test: Tensor<B, 2>,
    y_test: Tensor<B, 2>,
) {
    // Get model predictions
    let predictions = model.forward(x_test);

    // Compute Mean Squared Error (MSE)
    let mse = (predictions - y_test).powf(2.0).mean();


    // Print evaluation result
    println!("Mean Squared Error: {:?}", mse);

}
