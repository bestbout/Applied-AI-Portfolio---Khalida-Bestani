# Sentiment Analysis of Movie Reviews
The goal of this project is to analyze movie reviews and automatically determine whether the sentiment is **positive** or **negative**. This helps movie studios and streaming platforms understand audience opinions without manually reading thousands of reviews.

## Approach
1. **Dataset:** Used the IMDB Movie Review dataset containing 50,000 labeled reviews (25,000 positive, 25,000 negative).  
2. **Data Exploration:** Checked for balance, review length distributions, and common words using WordCloud visualizations.  
3. **Preprocessing:** Cleaned text by converting to lowercase, removing punctuation, and eliminating stopwords.  
4. **Feature Extraction:** Converted text into numerical features using **TF-IDF Vectorization** (top 5000 features).  
5. **Modeling:** Trained and compared two machine learning models:
   - **Logistic Regression** (baseline)
   - **Random Forest Classifier**
6. **Evaluation:** Compared models using **accuracy, precision, recall, classification reports, and confusion matrices**.

## Results
- **Logistic Regression:**  
  - Accuracy: 89.28%  
  - Precision & Recall: Balanced for both positive and negative reviews.  
  - Confusion: Misclassified some negative reviews as positive.  
- **Random Forest:**  
  - Accuracy: 85.09%  
  - Less consistent than Logistic Regression; struggled with sparse text data.  

## Key Findings 
- Logistic Regression performed better because TF-IDF features are **high-dimensional and sparse**, which suits linear models.  
- Positive reviews commonly include words like *"great", "love", "amazing"*.  
- Negative reviews commonly include words like *"bad", "boring", "waste"*.  
- The dataset is balanced, allowing the model to learn both classes effectively.  

## Technologies Used

- Python  
- Jupyter Notebook / Google Colab  
- Pandas, NumPy  
- Matplotlib, Seaborn, WordCloud  
- NLTK (for stopwords)  
- Scikit-learn (Logistic Regression, Random Forest, TF-IDF Vectorizer)

## How to Run
1. Open the notebook **Sentiment-Analysis-of-service.ipyn** in **Google Colab** or **Jupyter Notebook**.  
2. Ensure the **IMDB dataset CSV** is in the notebook directory or uploaded to Colab.  
3. Run each cell sequentially to see preprocessing, EDA, model training, and evaluation results.  


