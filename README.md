# 🧠 Linear Regression Model in Rust

## 📌 Project Overview
This project showcases a **simple yet powerful AI model for linear regression** built using **Rust** and the **Burn library (v0.16.0)**. The model predicts values based on the equation:

\[ y = 2x + 1 \]

By generating synthetic data with added noise, we simulate real-world conditions and train a model to make accurate predictions. The training process is optimized using **Mean Squared Error (MSE) loss**, ensuring a robust and efficient learning experience.

---
## 🚀 Why This Project?
- ✅ **Hands-on Machine Learning with Rust** – Explore AI in Rust, a fast and memory-safe language.
- ✅ **Practical Implementation** – Apply linear regression to real-world data.
- ✅ **Performance-Optimized** – Leverages the Burn framework for efficient computation.
- ✅ **Data Visualization** – Uses `textplots` to graphically display results in the terminal.

---
## 🛠 Setup Instructions

### **1️⃣ Install Rust**
Download and install Rust from [Rust's official website](https://www.rust-lang.org/tools/install):
```sh
rustc --version  # Verify installation
```

### **2️⃣ Install Rust Rover IDE**
Download Rust Rover from [JetBrains](https://www.jetbrains.com/rust/) and set up your project environment.

### **3️⃣ Clone the Repository**
```sh
git clone https://github.com/hlulanij/linear_regression_model.git
cd linear_regression_model
```

### **4️⃣ Install Dependencies**
Ensure the required dependencies are listed in `Cargo.toml`:
```toml
[dependencies]
burn = { version = "0.16.0", features = ["wgpu", "train"] }
burn-ndarray = "0.16.0"
rand = "0.9.0"
rgb = "0.8.50"
textplots = "0.8.6"
```
Then, run:
```sh
cargo build
```

### **5️⃣ Train & Run the Model**
To execute the linear regression model:
```sh
cargo run
```

---
## 📊 Features
✔ **Synthetic Data Generation** – Creates randomized (x, y) data pairs with noise  
✔ **Linear Regression Model** – Implements a single-layer model with optimized weights  
✔ **MSE Loss Function** – Tracks model performance during training  
✔ **Training & Evaluation** – Ensures the model converges to accurate predictions  
✔ **Graphical Representation** – Uses `textplots` to visualize training results  

---
## 📂 Project Structure
```
linear_regression_model/
├── src/
│   ├── main.rs       # Main program entry
│   ├── dataset.rs    # Generates training data
│   ├── model.rs      # Defines and trains the model
├── Cargo.toml        # Rust dependencies
├── README.md         # Project documentation
```

---
## 📝 License
This project is open-source and available under the **MIT License**.

---
## 📧 Contact & Contributions
Have questions or want to contribute? Reach out via **@hlulanij** on GitHub. Pull requests are welcome! 🎯

---

