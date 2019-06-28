import logging
from services import classification_service
from flask import request
logger = logging.getLogger("default")


def index():
    logger.info("Checking the flask scaffolding logger")
    return "Welcome to the flask scaffolding application"


def accuracy_of_test_data():
    logger.info("connecting to classifier.....")
    """
    TASKS: write the logic here to parse a json request
           and send the parsed parameters to the appropriate service.

           return a json response and an appropriate status code.
    """
    accuracy = classification_service.ClassificationService.train(self = 'training')
    return accuracy

def similar_dept():
    logger.info("suggesting similar department...")
    jd = request.form['job-description']
    recommendations = classification_service.ClassificationService.predict(self = "predicting", jd = jd)
    return recommendations
