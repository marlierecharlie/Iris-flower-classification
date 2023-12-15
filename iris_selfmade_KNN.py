from scipy.spatial import distance
from fastapi import FastAPI, HTTPException, Request, Form
from sklearn.metrics import accuracy_score as acs
import uvicorn
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import List

def eu(a, b):
	return distance.euclidean(a, b)

class KNN:
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist = eu(row, self.x_train[0])
		best_index = 0
		for i in range (1, len(self.x_train)):
			dist = eu(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clf = KNN()
clf.fit(x_train, y_train)

app = FastAPI()
@app.post("/submit")
async def pred(submit: IrisFeatures):
    try:
        # Map des indices aux noms de classe
        class_names = ['setosa', 'versicolor', 'virginica']

        # Convert received features to a list for prediction
        input_features = [submit.sepal_length, submit.sepal_width, submit.petal_length, submit.petal_width]

        # Perform prediction using your KNN model
        prediction_index = clf.predict([input_features])[0]

        # Obtenir le nom de la classe correspondante
        prediction_class = class_names[prediction_index]

        return prediction_class
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run("essai:app", host='127.0.0.1', port=8000, reload=True)
