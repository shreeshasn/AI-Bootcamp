import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# x-no of hours studies
x = np.array([[0.25],[0.5],[0.75],[1],[2],[3],[4],[5],[6],[7]])

# print(x.shape)

# y-pass (1) or fail(0)
y = np.array([0,0,0,0,1,1,1,1,1,1])

# print(y.shape)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
print("Data split into training and testing sets. \n")

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

print("Training the model... \n")

model = LogisticRegression()   #train
model.fit(x_train, y_train)
print("Model trained successfully. \n")

y_pred=model.predict(x_test)  #test
print("Predictions on test set: \n", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model: {:.2f}% \n".format(accuracy * 100))

# Making a prediction for a student who studies 4.5 hours
# hours_studied = float(input("Enter the number of hours studied by the student: \n"))
# prediction = model.predict([[hours_studied]])
# result = "Pass" if prediction[0] == 1 else "Fail"
# print(f"\nPrediction for a student who studies {hours_studied} hours: {result} \n")


hours = np.array([[4.5],[1.9],[0.3],[6.5],[2.2],[0.7],[5.5],[3.3],[0.1],[7.0]])
result = model.predict(hours)
res=0

print("Predictions for multiple students based on hours studied: \n")
for h,r in zip(hours,result):
    res = "Pass" if r == 1 else "Fail"
    print(f"Hours Studied: {h} - Prediction: {res}")