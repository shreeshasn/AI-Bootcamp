import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x = np.array([[0.25],[0.5],[0.75],[1],[2],[3],[4],[5],[6],[7]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test) 

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy} \n")

hours = np.array([[4.5],[1.9],[0.3],[6.5],[2.2],[0.7],[5.5],[3.3],[0.1],[7.0]])
result = model.predict(hours)
res=0

print("Predictions for unused data: \n")
for h,r in zip(hours,result):
    res = "Pass" if r == 1 else "Fail"
    print(f"Hours Studied: {h} - Prediction: {res}")