import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
scores = np.array([30,35,40,50,55,60,65,70,80])

model = LinearRegression()
model.fit(hours, scores)

prediction = model.predict([[5]])
print("Predicted score for 5 hours:", prediction[0])

plt.scatter(hours, scores)
plt.plot(hours, model.predict(hours))
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Student Score Prediction")
plt.show()
