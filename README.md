#  Customer Churn Prediction using Artificial Neural Networks (ANN)
# 1. Problem Statement
Customer churn — when clients stop doing business with a company — is a key challenge for subscription-based and service industries.  
The objective of this project is to **predict the likelihood of customer churn** using an **Artificial Neural Network (ANN)** model, enabling proactive retention strategies and improving long-term customer value.
**Goal:**  
Build and deploy an ANN model that predicts whether a customer will leave the bank based on demographic and behavioral data.

## 2. Exploratory Data Analysis (EDA) and Preprocessing
### **Key EDA Insights**
- **Age and Balance:** Older customers and those with higher balances showed higher churn rates.  
- **Geography:** Customers from Germany had a significantly higher churn rate than those from France or Spain.  
- **Activity Level:** Inactive members had a higher probability of churn.  
- **Gender:** Female customers showed a slightly higher churn tendency.
- **Encoding Categorical Variables:**
   - Applied **Label Encoding** to `Gender`.  
   - Applied **One-Hot Encoding** to `Geography`.
-  **Feature Scaling:** Standardized numerical variables using `StandardScaler`.
-  
## 3. Model Architecture and Training
### **Model Type**
A fully connected **Feedforward Artificial Neural Network (ANN)** built using **TensorFlow / Keras**.
## Training Details
Epochs: 100
Batch Size: 32
Optimizer: Adam
Loss: Binary Crossentropy
Callbacks: EarlyStopping for convergence

## 4. Evaluation Metrics
| Metric              | Value |
| Training Accuracy   |   86% |
| Test Accuracy       | 85.6% |
| Precision           |  0.82 |
| Recall              |  0.79 |
| F1-Score            |  0.80 |

## 5. Deployment and Impact
Deployment using Streamlit
The trained ANN model (model.h5) was deployed using a Streamlit app (app.py) for real-time predictions.
## Key Features of the App
- Interactive input fields for all customer parameters
- Live churn probability displayed with interpretation
- Uses saved preprocessing objects:
sc.pkl → Scaler
le.pkl → LabelEncoder
oh.pkl → OneHotEncoder
## Output Interpretation:
> 0.5: Customer likely to churn
≤ 0.5: Customer likely to stay

## Business Impact
- Enables proactive customer retention strategies.
- Helps marketing teams prioritize high-risk customers.
- Data-driven decision-making reduces churn and boosts revenue retention.

## Tech Stack
- Python: 3.x
- Libraries: TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Streamlit, Pickle
- Deployment: Streamlit Web App
- Environment: VS Code
- Version Control: Git + GitHub
