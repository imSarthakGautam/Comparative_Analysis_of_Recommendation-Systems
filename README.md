# Comparative Analysis of Recommendation-Systems
This repository consists of few Recommendation algorithms, which form the various Recommendation System.


Recommendations systems are used everywhere from online shopping to movies to music. The project is a implementation of the best recommendation model among some generally used recommendation models tested thoroughly by introducing user-item data to it.
The models that we have taken into account for this project are:
 - k Nearest Neighbours
 - Jaccard's similarity
 - Matrix Factorization
 - Dense Neural Network

These four different models are amongst the mostly used models in various Recommendation system.
Throughout the course of this project, Comparative Analysis of Recommendation Systems to recommend books, various machine learning models have been implemented, trained and tested in order to provide best possible resultant recommendation for users.
In order to effectively execute the project objectives alongside a well maintained large
database, the fine tuning of various models is essential. The major steps involved in this are:
- Dataset Collection
- Pre-processing the data
- Model Training
- Testing
- Post Processing
- Verification and Validation

### Dataset Collection
The dataset is collected from Kaggle. This dataset comprising of three tables for users,
books and ratings was compiled by Cai-Nicolas Ziegler. The dataset was formed on the
basis of books read by users and ratings provided by them on Amazon and the online
data for books from Amazon along with user ratings and users who bought them. The
dataset was originally in .CSV format


## Model training

### KNN
k-Nearest Neighbors (KNN) is a type of supervised learning algorithm used for regression tasks which predicts the label or value of a data point based on the average of its k-nearest neighbors in the feature space. Here for implementation of KNN we have utilized ’NearestNeighbours’ class of scikit-learn library, which enables to find the k-neighbours for given book in feature space Choice of Hyperparameter k: k represents the number of neighbors considered for each prediction. The fine tuning of hyperparameter is very essential as smaller too small k can lead to overfitting, while too large k can result in underfitting. After iterative testing the value of k was choosen to be 7.

### Matrix Factorization
Matrix Factorization is a collaborative filtering technique commonly used in book recommendation systems. It involves decomposing the user-item interaction matrix into two lower-dimensional matrices representing users and items with the aim that when reconstructed back again it captures hidden pattern or the latent factors that help to predict missing values and generate personalized recommendations. For the implemendation and training of Matrix Factorization first of all a sparse matrix of user-item i.e UserID-BookId was formed which is also commonly known as pivot column. Then the ’TruncatedSVD’ class was used to perform dimensionality reduction on the transpose of this pivot column which was then fitted into Matrix Factorization model whose resultant would provide recommendations.

### Neural Network
A Neural Network is a machine learning model inspired by the structure and functioning of the human brain. It consists of layers of interconnected nodes, or neurons, organized into an architecture that allows the model to learn patterns and relationships in data. Dense layers, also known as fully connected layers, are a type of layer in a neural network where each neuron is connected to every neuron in the previous and subsequent layers. These layers are often used to capture complex patterns and relationships in the data. Optimization techniques are algorithms used to adjust the model’s parameters during training in order to minimize the error or loss. Adam optimizer has been used which is a popular optimization algorithm that combines the benefits of both the AdaGrad and RMSProp algorithms. Mean Squared Error (MSE) has been used as the loss function. The optimizer is responsible for updating the model’s weights during training, while the loss function quantifies how well the model is performing by measuring the difference between the predicted values and the actual values. The model was trained for a total of 50 epochs.

### Jaccard’s Similarity
Jaccard’s Similarity is the measusre of similarity between two sets. A user-item interaction matrix has been constructed where rows represent users and columns represent books and the entries in the matrix represent user interactions or ratings. A user-user similarity matrix has also been constructed based on the similarity between users. Jaccard’s Similarity has been used to measure this similarity. The Jaccard’s Similarity between two users is calculated based on their interactions with items. It measures the similarity in terms of the items they have interacted with via the rating values. The formula for Jaccard’s Similarity is applied to the sets of items each user has interacted with. It computes the ratio of the number of common items to the total number of unique items across both users. Once the user-user similarity matrix had been constructed using Jaccard’s Similarity or other similarity measures, it provided a numerical representation of how similar each pair of users is. To generate recommendations for a target user, the system identifies users who are most similar to the target user based on the user-user similarity matrix. Items liked or interacted with by similar users but not by the target user were then recommended to the target user. This assumes that users with similar preferences will likely appreciate items that the target user has not yet interacted with. For recommendation system's entire procedures, we have selected Books as our units i.e Recommendation of books as the application/resource upon which models were designated to work. The dataset used for testing is the Book Crossings Dataset which has been collected from Kaggle and has 1.1 million ratings of 230,000 books by 90,000 users. During testing, the most balanced model among the comparisions was found to be the kNN model. Scikit learn’s NearestNeighbors model is a simple yet effective regression model that can be used for recommendations. After analysing, cleaning and formatting the dataset, it was fed into the model based on which the recommendations are generated. This model was compared against the Scikit learn’s TruncatedSVD which is a Matrix Factorization model’s Singular Value Decomposition model, Jaccard Similarity and a Dense Neural Network. During testing the kNN model was found to be quite goad at mainitaining good scores throughout the board with precision of 0.8045, recall of 0.7503 and F1-score of 0.7410. The other models though extremely better performing in one metric, lagged considerably in the other metrics.

The performance of our models has also been evaluated in the context of their strengths and weaknesses. The kNN model, while simple and easy to implement, tends to identify a large number of relevant items mostly popular items and might struggle to recommend users with unique tastes. On the other hand, the Matrix Factorization model, while more complex, could uncover latent factors that captures underlying structures in the dataset and provide more personalized recommendations. Additionally, handling missing values and incorporating side information (e.g. user demographics, book attributes) can further improve the accuracy of recommendations. Jaccard’s Similarity and Neural Network seem to work well when there are larger sets of data to be trained with showing capable relationship determining abilities but seemed lacking for the amount of data that was fed into the model. Moving forward, potential research explore hybrid approaches that encompasses the strengths of both models for further improved recommendation performance.
