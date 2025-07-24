
# 🧠 Day 13: Intent Recognition in Hinglish  
> **Project Type:** Machine Learning  
> **Language:** Hinglish (Hindi + English)  
> **Toolkits Used:** TensorFlow, NLP, Scikit-learn, NLTK  
> **Notebook:** `Day13_Intent_Recognition_Hinglish.ipynb`

## 📌 Project Overview
This project demonstrates a **Hinglish Intent Recognition** system as part of a daily deep learning series. The aim is to classify user intent based on mixed Hindi-English (Hinglish) input — which is common in real-world conversational AI applications in India.

The notebook showcases:
- Preprocessing of Hinglish text
- Label encoding of intents
- Text vectorization
- Model training using deep learning (Dense NN)
- Model evaluation with accuracy & loss metrics

## 🎯 Objectives
- ✅ Build a dataset containing Hinglish phrases mapped to intents  
- ✅ Preprocess mixed-language data using NLTK and regular expressions  
- ✅ Encode intents using label encoding  
- ✅ Train a simple deep neural network for classification  
- ✅ Test and validate predictions

## 🧰 Technologies & Libraries Used

| Tool/Library | Purpose |
|--------------|---------|
| **Python 3.x** | Programming Language |
| **TensorFlow/Keras** | Model Building |
| **Scikit-learn** | Label Encoding, Data Splitting |
| **NLTK** | Text Preprocessing |
| **NumPy, Pandas** | Data Handling |
| **Matplotlib** | Visualization |

## 📁 Project Structure

```
📦Intent_Recognition_Hinglish
 ┣ 📜Day13_Intent_Recognition_Hinglish.ipynb
 ┣ 📜README.md
 ┣ 📁data
 ┃ ┗ 📜intent_dataset.json
 ┗ 📁outputs
   ┗ 📜model.h5 (Optional)
```

## 🧪 How to Run

### 🔧 Requirements
Install the required packages:

```bash
pip install numpy pandas scikit-learn tensorflow nltk
```

### ▶️ Run the Notebook

1. Clone the repository:
    ```bash
    git clone https://github.com/yashpalsince2004/Intent_Recognition_Hinglish.git
    cd Intent_Recognition_Hinglish
    ```

2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook Day13_Intent_Recognition_Hinglish.ipynb
    ```

## 🗃️ Dataset Description
The dataset consists of a list of user queries mapped to a label called `intent`.  
Example:

```json
{
  "text": "hello kaise ho?",
  "intent": "greeting"
}
```

The dataset is loaded from JSON and parsed into pandas DataFrame for training and preprocessing.

## 🧠 Model Architecture
A basic **feedforward neural network**:

- Input Layer: Tokenized and vectorized text input  
- Hidden Layers: Dense layers with ReLU  
- Output Layer: Softmax for multi-class classification

Loss function: `sparse_categorical_crossentropy`  
Optimizer: `adam`  
Metric: `accuracy`

## 📊 Results

- **Training Accuracy**: ~90%+ (depending on data split)
- **Loss Graphs**: Plotted using Matplotlib for training vs validation performance

## 🔮 Example Predictions

| Input                      | Predicted Intent |
|---------------------------|------------------|
| `"hello"`                 | `greeting`       |
| `"mujhe movie dekhni hai"` | `entertainment`  |
| `"thanks bhai"`           | `gratitude`      |

## 📌 Future Improvements

- Expand the Hinglish dataset  
- Add attention-based models (e.g., LSTM or Transformer)  
- Deploy the model as a REST API  
- Integrate into a chatbot frontend (e.g., Flutter or React)

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is open-source and available under the MIT License.

## 🙋‍♂️ Author
**Yash Pal**  
🎓 Final Year B.Tech – AI & ML  
📬 [LinkedIn](https://www.linkedin.com/in/yash-pal-since2004) • [GitHub](https://https://github.com/yashpalsince2004)
