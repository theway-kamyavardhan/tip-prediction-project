```markdown
# 💡 Tip Prediction using Linear Regression

This project demonstrates a basic machine learning workflow using **Linear Regression** to predict restaurant tips based on customer details such as total bill, gender, smoking status, day, time, and party size.

---

## 📁 Project Structure

```

tip\_prediction\_project/
├── tip\_predictor.py       # Main Python script
├── tips.csv               # Dataset file
├── README.md              # This file

````



## 📊 Dataset

The dataset (`tips.csv`) contains restaurant bill information, including:

- `total_bill`: Total bill (in dollars)
- `tip`: Tip given
- `sex`: Customer gender
- `smoker`: Whether the customer is a smoker
- `day`: Day of the week
- `time`: Lunch or Dinner
- `size`: Party size (number of people)



## 🔧 Features Used

| Feature       | Description              |
|---------------|---------------------------|
| `total_bill`  | Total cost of the meal     |
| `sex`         | Gender (encoded)           |
| `smoker`      | Smoker or not (encoded)    |
| `day`         | Day of week (encoded)      |
| `time`        | Time of day (encoded)      |
| `size`        | Number of people at table  |

---

## 🧠 What the Script Does

1. Loads and prints the dataset
2. Encodes categorical columns (`sex`, `smoker`, `day`, `time`)
3. Scales features using `StandardScaler`
4. Splits the data (70% training, 30% testing)
5. Trains a **Linear Regression** model
6. Predicts tips on the test set
7. Evaluates using R² Score and Mean Squared Error
8. Plots actual vs predicted tips

---

## ▶️ How to Run

### 🧩 Prerequisites

Make sure you have Python installed. You can install the required libraries with:

```bash
pip install pandas numpy scikit-learn matplotlib
````

### ▶️ Run the Script

Navigate to the project folder and run:

```bash
python tip_predictor.py
```

> 💡 Make sure `tips.csv` is in the same folder as your Python file.

---

## 📈 Example Output

* **Model R² Score**: `0.44` *(example)*
* **Mean Squared Error**: `0.98` *(example)*
* A scatter plot comparing actual vs predicted tips

---

## 📌 Notes

* The categorical data is encoded using **LabelEncoder**
* Feature scaling is applied using **StandardScaler**
* This is a basic ML regression project meant for learning purposes

---

## 📚 License

This project is open-source and free to use under the MIT License.

---

## 👨‍💻 Author

Made by **Kamyavardhan Dave**
Connect with me on [GitHub](https://github.com/theway-kamyavardhan) or [LinkedIn](https://www.linkedin.com/in/kamyavardhan)

```

