import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk


df = pd.read_csv("D:\\projects\\Churn\\churn_dataset.csv")

#print(df.head())

missing_values = df.isnull()
# print("Missing Values:")
# print(missing_values)

summary_stats = df.describe()

# plt.figure(figsize=(12, 6))
# sns.histplot(df['tenure'], bins=30, kde=True)
# plt.title('Distribution of Tenure')
# plt.xlabel('Tenure (months)')
# plt.ylabel('Frequency')
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.countplot(x='Contract', hue='Churn', data=df)
# plt.title('Churn by Contract Type')
# plt.xlabel('Contract Type')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check the data type of 'TotalCharges' after conversion
# print(df['TotalCharges'].dtype)

df['TotalChargesPerMonth'] = df['TotalCharges'] / df['tenure']

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalChargesPerMonth'] = df['TotalCharges'] / df['tenure']

#Updated DataFrame
# print(df.head())

X = df.drop('Churn', axis=1)  
y = df['Churn']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)


#one-hot encoding
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

#numerical columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

#missing values
imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

# one hot encoding for categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

# one hot encoded features for numerical features
X_train_processed = np.hstack((X_train_encoded, X_train[numerical_cols]))
X_test_processed = np.hstack((X_test_encoded, X_test[numerical_cols]))

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train_processed, y_train)

y_pred = clf.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred)
# print("Classification Report:\n", classification_rep)

conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)








# number of customers who churned with pie chart and percentage


# churn_counts = df['Churn'].value_counts()

# plt.figure(figsize=(6, 6))
# plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title('Churn Distribution in the Entire Dataset')
# plt.show()

# percentage_churn = (churn_counts['Yes'] / churn_counts.sum()) * 100
# percentage_not_churn = (churn_counts['No'] / churn_counts.sum()) * 100

# print("Percentage of Customers Who Churned:", percentage_churn, "%")
# print("Percentage of Customers Who Did Not Churn:", percentage_not_churn, "%")











#by entering customer ID


# predictcustomer_id = input("Enter Customer ID: ")

# customer_data = df[df['customerID'] == predictcustomer_id].copy()

# # Preprocess the customer's data
# customer_data = customer_data.drop(['customerID', 'Churn'], axis=1) 
# customer_data[numerical_cols] = imputer.transform(customer_data[numerical_cols])
# customer_data_encoded = encoder.transform(customer_data[categorical_cols])
# customer_data_processed = np.hstack((customer_data_encoded, customer_data[numerical_cols]))

# # Use the trained model to make predictions for the customer
# customer_prediction = clf.predict(customer_data_processed)

# # Print the prediction
# if customer_prediction[0] == 'Yes':
#     print("Customer with ID", predictcustomer_id, "is predicted to churn.")
# else:
#     print("Customer with ID", predictcustomer_id, "is predicted not to churn.")










#using window


def predict_churn():
    customer_id = entry_customer_id.get()

    customer_data = df[df['customerID'] == customer_id].copy()
    customer_data = customer_data.drop(['customerID', 'Churn'], axis=1)  
    customer_data[numerical_cols] = imputer.transform(customer_data[numerical_cols])
    customer_data_encoded = encoder.transform(customer_data[categorical_cols])
    customer_data_processed = np.hstack((customer_data_encoded, customer_data[numerical_cols]))

    customer_prediction = clf.predict(customer_data_processed)

    info_text = "Customer Information:\n"
    for col, value in customer_data.iloc[0].items():
            info_text += "{}: {}\n".format(col, value)
            info_text += "\n"
    text_widget = tk.Text(window, wrap=tk.WORD, height=20, width=40) 
    text_widget.tag_configure("custom_font", font=("Helvetica", 12))
    text_widget.tag_configure("custom_color", foreground="blue") 
    text_widget.insert(tk.END, info_text)
    text_widget.place(x=350, y=300)
    text_widget.tag_configure("left", justify="left")

    if customer_prediction[0] == 'Yes':
        prediction_result.set("Customer with ID {} is predicted to churn.".format(customer_id))
    else:
        prediction_result.set("Customer with ID {} is predicted not to churn.".format(customer_id))
        
window = tk.Tk()
window.geometry("854x480")
window.title("Customer Churn Prediction")

bg_image = Image.open("bg.png")
bg_image = ImageTk.PhotoImage(bg_image)

canvas = tk.Canvas(window, width=400, height=300)
canvas.place(x=0, y=0, relwidth=1, relheight=1)

canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)

label = canvas.create_text(200, 150, text="Enter Customer ID: ", font=("Helvetica", 20, "bold"), fill="white", anchor=tk.CENTER)

def clear_entry(event):
    if entry_customer_id.get() == "Enter here!":
        entry_customer_id.delete(0, tk.END)

entry_customer_id = tk.Entry(window, width=25)
entry_customer_id.place(x=350, y=140)
entry_customer_id.insert(0, "Enter here!")

entry_customer_id.bind("<FocusIn>", clear_entry)

predict_button = tk.Button(window, text="Predict Churn", command=predict_churn, width=15, height=2)
predict_button.place(x=350, y=180)

prediction_result = tk.StringVar()
result_label = tk.Label(window, textvariable=prediction_result, font=("Helvetica", 14), fg="black")
result_label.place(x=250, y=250)

window.mainloop()