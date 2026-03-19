---
title: House Price Predictor
emoji: 🏠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# 🏠 House Price Prediction System

**Prediction des Prix Immobiliers** - A complete Machine Learning web application for predicting house prices using Random Forest Regression.

## 📋 Project Overview

This project demonstrates a full ML pipeline including:
- **Data Generation**: Synthetic dataset with 1000 samples and 7 features
- **Model Training**: Random Forest Regressor with 100 estimators
- **Web Interface**: Interactive Gradio application
- **Containerization**: Docker deployment
- **Cloud Deployment**: HuggingFace Spaces

### Features
- 🎯 Real-time price predictions
- 📊 Three interactive visualizations:
  1. Feature importance analysis
  2. Input values summary
  3. Prediction with 95% confidence interval
- 🖥️ User-friendly web interface
- 🐳 Fully containerized application

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Tp2_docker
```

2. **Create virtual environment**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python train_model.py
```

5. **Run the application**
```bash
python app.py
```

Visit `http://127.0.0.1:7860` in your browser.

### Docker Deployment

1. **Build the image**
```bash
docker build -t house-price-predictor:latest .
```

2. **Run the container**
```bash
docker run -p 7860:7860 house-price-predictor:latest
```

3. **Push to Docker Hub** (optional)
```bash
docker tag house-price-predictor:latest <username>/house-price-predictor:latest
docker push <username>/house-price-predictor:latest
```

## 📊 Dataset Features

The model uses 7 features to predict house prices:

| Feature | Description | Range |
|---------|-------------|-------|
| **square_feet** | Total living area | 800 - 4,000 sq ft |
| **bedrooms** | Number of bedrooms | 1 - 5 |
| **bathrooms** | Number of bathrooms | 1 - 4 |
| **age_years** | Age of the property | 0 - 50 years |
| **lot_size** | Total land area | 2,000 - 10,000 sq ft |
| **garage_spaces** | Parking capacity | 0 - 3 |
| **neighborhood_score** | Quality rating | 1 - 10 |

## 🧠 Model Architecture

- **Algorithm**: Random Forest Regressor
- **Estimators**: 100 trees
- **Max Depth**: 15
- **Preprocessing**: StandardScaler normalization
- **Train/Test Split**: 80/20
- **Evaluation Metrics**: MAE, R²

## 📁 Project Structure

```
Tp2_docker/
├── app.py                  # Gradio web application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── .dockerignore          # Docker ignore rules
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── model.pkl              # Trained model
├── scaler.pkl             # Feature scaler
├── feature_names.pkl      # Feature names
├── data/                  # Dataset storage
│   └── house_prices.csv
├── models/                # Model artifacts (backup)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── plots/                 # Generated visualizations
    ├── feature_importance.png
    ├── input_summary.png
    └── prediction.png
```

## 🧪 Testing

Run the test suite with:

```bash
python -m pytest test_train.py -v
```

### Test coverage — 11 tests across 4 classes

| Class | What it checks |
| ------- | --------------- |
| `TestGenerateSyntheticData` | Shape (1000×8), column names, feature ranges, no nulls, positive prices, seed reproducibility |
| `TestScalerLeakage` | `StandardScaler` is fit only on training data (no data leakage) |
| `TestModelTraining` | Prediction output shape and test R² ≥ 0.85 |
| `TestArtifactPersistence` | End-to-end: train → pickle artifacts → reload → predict on a sample input |

> **Note**: `pytest` is not in `requirements.txt` (it's a dev dependency). Install it with `pip install pytest` before running.

## 🔧 Technical Stack

- **Python**: 3.11
- **ML Framework**: scikit-learn 1.3.0
- **Web Framework**: Gradio 4.19.2
- **Data Processing**: pandas 2.0.3, NumPy 1.24.3
- **Visualization**: matplotlib 3.7.2, seaborn 0.12.2
- **Containerization**: Docker
- **Deployment**: HuggingFace Spaces

## 📖 Comprehension Questions & Answers

### Gradio Questions

**Q1: Why return file paths instead of images directly in Gradio?**
- Gradio efficiently handles image display using file paths
- Reduces memory usage for large images
- Enables caching and faster reload
- Better performance in web applications

**Q2: Difference between `gr.Row()` and `gr.Column()`?**
- `gr.Row()`: Arranges components horizontally (side by side)
- `gr.Column()`: Arranges components vertically (stacked)
- Used together to create complex layouts

**Q3: How does Gradio update the interface on button click?**
- Uses the `.click()` event handler
- Connects button to function via `fn` parameter
- Maps `inputs` to function arguments
- Updates `outputs` with returned values
- Automatically refreshes UI components

### Docker Questions

**Q4: Difference between image and container?**
- **Image**: Read-only template with application code, dependencies, and configuration (blueprint)
- **Container**: Running instance of an image (actual execution environment)
- Analogy: Image is like a class, container is like an object

**Q5: What happens if port 7860 is not exposed?**
- The application runs inside the container but is inaccessible from outside
- No external connections can reach the Gradio server
- `EXPOSE` documents the port but doesn't publish it
- Must use `-p 7860:7860` when running to map ports

### Git Questions

**Q6: Difference between `git add` and `git commit`?**
- `git add`: Stages changes (adds to staging area for next commit)
- `git commit`: Saves staged changes to local repository with a message
- Workflow: modify → add → commit

**Q7: Purpose of `.gitignore`?**
- Prevents specified files/folders from being tracked by Git
- Excludes temporary files, dependencies, secrets, and build artifacts
- Keeps repository clean and secure
- Reduces repository size

**Q8: What does `-u` do in `git push -u origin main`?**
- Sets upstream tracking reference
- Links local branch to remote branch
- Allows future `git push` without specifying remote/branch
- Short for `--set-upstream`

### Machine Learning Questions

**Q9: Why normalize features with `StandardScaler`?**
- Makes features have mean=0 and std=1
- Prevents features with larger ranges from dominating
- Improves model convergence and performance
- Essential for distance-based algorithms
- Random Forest is less sensitive but normalization still helps

**Q10: What is 95% confidence interval and how to compute it?**
- Range where we're 95% confident the true value lies
- Computed as: prediction ± 1.96 × standard_error
- Uses predictions from all trees in Random Forest
- 1.96 comes from the z-score for 95% confidence in normal distribution

**Q11: Why split data into training and test sets?**
- **Training set**: Used to train the model
- **Test set**: Evaluates model on unseen data
- Prevents overfitting
- Provides unbiased performance estimate
- Standard split: 80/20 or 70/30

## 🎯 Key Performance Metrics

After training, the model achieves:
- **Test MAE**: ~$25,000 - $35,000
- **Test R² Score**: ~0.90 - 0.95

## 🔗 Links

- **GitHub Repository**: [[Github-House-price-predictor](https://github.com/SamiMalek10/House-Price-Prediction)]
- **Docker Hub**: [[Docker-House-price-predictor](https://hub.docker.com/repository/docker/samimlk/house-price-prediction/)]
- **HuggingFace Space**: [[Hugginface-House-price-predictor](https://huggingface.co/spaces/sameeeeuaytehk15/house-price-predictor)]

## 👨‍💻 Author

**TP2: Machine Learning Project**
- Date: November 17, 2025
- Course: Machine Learning & DevOps

## 📝 License

MIT License - Feel free to use and modify for educational purposes.

## 🙏 Acknowledgments

- scikit-learn for the ML framework
- Gradio for the intuitive web interface
- HuggingFace for hosting capabilities
