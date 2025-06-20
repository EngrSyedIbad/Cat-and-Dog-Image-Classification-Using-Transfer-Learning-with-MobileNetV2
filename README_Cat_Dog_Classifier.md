# 🐱🐶 Cat and Dog Image Classification using Transfer Learning (MobileNetV2)

This project implements an efficient and accurate image classifier to distinguish between cats and dogs using **MobileNetV2**, a lightweight deep learning architecture, via **transfer learning** in TensorFlow/Keras.

---

## 🧠 Key Highlights

- ✅ Transfer learning with pre-trained **MobileNetV2**
- 📷 Binary image classification: **Cats vs. Dogs**
- 🔄 Data augmentation for better generalization
- 📉 Training vs. Validation curves to monitor learning
- 📁 Google Colab-compatible with GPU support

---

## 📁 Project Structure

```plaintext
📄 Cat_and_Dog_Image_Classification_Using_Transfer_Learning_with_MobileNetV2.ipynb
📂 images/     ← Optional: store sample cat/dog images for README
📂 models/     ← Optional: save trained models
```

---

## 🚀 Model Architecture

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze the base model

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,  # Assumes defined earlier
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

## 📈 Performance Summary

| Metric           | Result (Example)     |
|------------------|----------------------|
| Training Accuracy| ~97%                 |
| Validation Accuracy | ~95%              |
| Loss Trend       | Smooth convergence   |
| Final Model Size | Lightweight          |

---

## ✅ Sample Results

| Image | Prediction |
|-------|------------|
| ![](images/sample_cat.jpg) | 🐱 Cat |
| ![](images/sample_dog.jpg) | 🐶 Dog |

> Add sample images in the `images/` folder or update the paths.

---

## 🛠 Setup Requirements

- Python ≥ 3.7  
- TensorFlow ≥ 2.8  
- Matplotlib, NumPy, Pandas  
- Run on [Google Colab](https://colab.research.google.com/) for GPU acceleration

---

## 🧪 How to Use

1. Open the notebook in [Colab](https://colab.research.google.com/)
2. Mount your Google Drive to access datasets:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload your dataset (Cats and Dogs folders).
4. Modify paths in the notebook.
5. Run all cells sequentially to train and evaluate the model.

---

## 🔍 Dataset

- Recommended: [Kaggle Cats vs. Dogs Dataset](https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats)

> Tip: Use a smaller subset for testing or split with validation folders.

---

## 🧯 Troubleshooting Common Issues

| Problem | Fix |
|--------|-----|
| ⚠️ Out of Memory | Reduce image size, batch size, or use generators |
| 🧠 Overfitting | Increase dropout, use augmentation |
| ❌ Low Accuracy | Unfreeze top base layers for fine-tuning |

---

## 📜 License

MIT License — free to use, share, and improve.