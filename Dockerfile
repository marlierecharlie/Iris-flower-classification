

# python base image in the container from Docker Hub
FROM python:3.10.6

# copy files to the /app folder in the container
COPY iris_selfmade_KNN.py app/iris_selfmade_KNN.py
COPY test_knn.py app/test_knn.py
COPY Pipfile /app/Pipfile
COPY Pipfile.lock app/Pipfile.lock


# set the working directory in the container to be /app
WORKDIR /app


# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile


# execute the command python main.py (in the WORKDIR) to start the app
CMD uvicorn iris_selfmade_KNN:app --host 0.0.0.0 --port $PORT
