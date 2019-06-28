from apis import views


api_urls = [
    ("/", views.index, ["GET"], "flask scaffolding index url"),
    ("/train", views.accuracy_of_test_data, ["GET"], "Accuracy of classifier.." ),
    ("/predict", views.similar_dept, ["POST"], "Response for user input")
]

other_urls = []

all_urls = api_urls + other_urls
