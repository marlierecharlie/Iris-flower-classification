#uvicorn fast:app –reload
gunicorn -w 4 -k uvicorn.workers.UvicornWorker iris_selfmade_KNN:app
