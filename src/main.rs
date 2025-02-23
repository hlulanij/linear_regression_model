mod dataset;
mod model;
mod train;
mod evaluate;

use evaluate::evaluate_model;
use model::LinearRegressionModel;
use burn::prelude::*;
use burn_ndarray::{NdArray, NdArrayDevice}; // Correct import for the backend


fn main() {
    // Create a device for NdArray backend
    let device = NdArrayDevice::default();

    let (x_values, y_values) = dataset::generate_dataset();
    let device = burn_ndarray::NdArrayDevice::default(); // ✅ Ensure `device` is used

    // Convert 1D Vec<f32> into 2D Vec<Vec<f32>>
    let x_values_flat: Vec<Vec<f32>> = x_values.iter().map(|&x| vec![x]).collect();
    let y_values_flat: Vec<Vec<f32>> = y_values.iter().map(|&y| vec![y]).collect();

    // ✅ Define x_tensor and y_tensor properly
    let x_tensor = Tensor::<NdArray<f32>, 2>::from_floats(&x_values_flat[..], &device);
    let y_tensor = Tensor::<NdArray<f32>, 2>::from_floats(&y_values_flat[..], &device);



    // Initialize model
    let model = LinearRegressionModel::<NdArray<f32>>::new(&device);

    // Forward pass
    let predictions = model.forward(x_tensor);

    // Create tensors
    let x_tensor = Tensor::<NdArray<f32>, 2>::from_floats(&x_values_flat, &device);
    let y_tensor = Tensor::<NdArray<f32>, 2>::from_floats(&y_values_flat, &device);

    // Print predictions
    println!("x_tensor: {:?}", x_tensor);
    println!("y_tensor: {:?}", y_tensor);

    // ✅ Call train function to use x_tensor
    let model = train::train_model(x_tensor.clone(), y_tensor.clone(), &device);

    // ✅ Ensure predictions are used
    let predictions = model.forward(x_tensor);
    println!("Predictions: {:?}", predictions);


}
