# Deep_NLP_chatbot_queries

# Dataset
Sheet_1.csv contains 80 user responses, in the response_text column, to a therapy chatbot. User responded. If a response is 'not flagged', the user can continue talking to the bot. If it is 'flagged', the user is referred to help.

# Model
I have made the infamous bag of words model. 

# Processing the data
I have preprocessed the data using the open source NLTK library and the 're' library.After the processing and cleaning the text, I have used the count vectorizer library for creating th bag of words model. 

# Training the dataset
I have split the training set and the test test using the scikitlearn library.I have taken 20% of the dataset as the test set. After spitting ,the data is scaled using the StandardScalar library. Then I have built an ANN model using keras(tensorflow backend).There is one input layer, 5 hidden layers and an output layer. The first 6 layers are tarined using the 'relu' activation function. The last layer is trained using the 'sigmoid' activation function.I have then evaluated the model.
