use rand::Rng; // Keep this
use rand::thread_rng; // Add this

pub fn generate_dataset() -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let num_samples = 100;
    let mut x_values = Vec::new();
    let mut y_values = Vec::new();

    for _ in 0..num_samples {
        let x: f32 = rng.gen_range(0.0..10.0);
        let y = 2.0 * x + 3.0; // Example linear function
        x_values.push(x);
        y_values.push(y);
    }

    (x_values, y_values)
}
