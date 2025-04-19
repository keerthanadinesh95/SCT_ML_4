# SCT_ML_4
Task 4: Hand Gesture Recognition — LeapGestRecog Dataset
SCT_ML_4 — SkillCraft Technology | Machine Learning Internship

💡 Project Overview
In this project, I developed a Hand Gesture Recognition Model using the LeapGestRecog Dataset from Kaggle. This dataset simulates hand gestures captured using the Leap Motion sensor, which can be used for touchless interfaces — a real-world application of computer vision and human-computer interaction!

The goal was to build a model that accurately classifies various hand gestures based on image data using deep learning techniques.

🗂️ Project Highlights
📚 Dataset: LeapGestRecog (Kaggle)

🧠 Model: Convolutional Neural Network (CNN) using Keras

🔍 Task: Multiclass Hand Gesture Classification

📊 Evaluation Metrics: Accuracy, Loss Graphs, and Confusion Matrix
## 📊 Accuracy & Loss Graphs
![Accuracy & Loss](/accuracy_loss_plot.png)

## 📌 Confusion Matrix
![Confusion Matrix](/confusion_matrix.png)


🔥 Workflow Summary
Data Preprocessing:

Loaded grayscale images.

Resized for uniformity.

Applied normalization for optimized model training.

Model Building:

Designed a CNN architecture to extract visual features.

Employed ReLU activation and Softmax for multi-class output.

Optimized using the Adam optimizer and Categorical Crossentropy loss.

Model Training & Evaluation:

Trained the model on the LeapGestRecog dataset.

Plotted Accuracy & Loss Curves for training and validation.

Evaluated the final model using a Confusion Matrix to visualize classification performance.

📈 Performance Insights
Achieved consistent accuracy in both training and validation sets.

Loss reduced smoothly, indicating successful learning.

The confusion matrix displayed strong classification capability across all gesture categories.

💾 How to Run the Project
Clone the repository:
git clone https://github.com/keerthanadinesh95/SCT_ML_4.git

Install the dependencies:
pip install -r requirements.txt

Run the Jupyter Notebook:
jupyter notebook Task4.ipynb

📌 Key Learnings
Deepened practical knowledge in image classification using CNNs.

Enhanced understanding of gesture recognition systems.

Improved ability to visualize model performance using confusion matrices and metric graphs.

✅ Task 4 Successfully Completed!
Looking forward to applying this knowledge to even more complex Computer Vision problems! 🚀

📌 Author
Keerthana Dinesh
LinkedIn: @keerthanadinesh95


