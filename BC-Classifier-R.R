rm(list = ls())
Sys.setenv(TZ='UTC')  # Sets the timezone to Coordinated Universal Time (UTC)
install.packages("keras")
install.packages("deepnet")
install.packages("mlbench")
packages <- c("deepnet",       
              "mlbench",
              "caret"
)
# load packages
invisible(lapply(packages, library, character.only = TRUE))

rm(list = ls())
data("BreastCancer") 

# Clean off rows with missing data 
BreastCancer = BreastCancer[which(complete.cases(BreastCancer) 
                                  == TRUE),] 
head(BreastCancer) 

names(BreastCancer)
y = as.matrix(BreastCancer[, 11]) 
y[which(y == "benign")] = 0 
y[which(y == "malignant")] = 1 
y = as.numeric(y) 
x = as.numeric(as.matrix(BreastCancer[, 2:10])) 
x = matrix(as.numeric(x), ncol = 9)
# Applying nn.train() function 
nn <- nn.train(x, y, hidden = c(7), activation = "sigm")
nn_t <- nn.train(x, y, hidden = c(7), activation = "tanh")
yy_t = nn.predict(nn_t, x)
yy = nn.predict(nn, x) 
print(head(yy))
print(head(yy_t))
# Convert the predictions to binary
yhat = matrix(0,length(yy), 1) 
yhat[which(yy > mean(yy))] = 1 
yhat[which(yy <= mean(yy))] = 0 
# Applying table() function 
cm = table(y, yhat) 
print(cm)

accuracy <- sum(diag(cm))/sum(cm)
print(paste("Accuracy: ", accuracy))


# Load necessary library
library(class)

# Your existing preprocessing steps
data("BreastCancer")
BreastCancer <- na.omit(BreastCancer)
BreastCancer$Class <- ifelse(BreastCancer$Class == "benign", 0, 1)

# Set the seed for reproducibility
set.seed(123)

# Create a stratified 70-30 split
trainIndex <- createDataPartition(BreastCancer$Class, p = 0.7, list = FALSE)
train_set <- BreastCancer[trainIndex, ]
test_set <- BreastCancer[-trainIndex, ]

# Train the k-NN model (let's use k = 3 for this example)
model <- knn(train = train_set[-ncol(train_set)], test = test_set[-ncol(test_set)], cl = train_set$Class, k = 5)

# The predictions are directly obtained from the model
predictions <- as.numeric(as.character(model))

# Evaluate the model performance
cm_knn <- confusionMatrix(table(predictions, test_set$Class))

print(cm_knn)


accuracy <- cm_knn$overall['Accuracy']
print(accuracy)

# Load necessary library
library(class)
library(caret)

# Your existing preprocessing steps
data("BreastCancer")
BreastCancer <- na.omit(BreastCancer)
BreastCancer$Class <- ifelse(BreastCancer$Class == "benign", 0, 1)

# Set the seed for reproducibility
set.seed(123)

# Create a stratified 70-30 split
trainIndex <- createDataPartition(BreastCancer$Class, p = 0.7, list = FALSE)
train_set <- BreastCancer[trainIndex, ]
test_set <- BreastCancer[-trainIndex, ]

# Select a subset of features, for example: first 5 features
selected_features <- c(1:5)

# Train the k-NN model using the selected features (let's use k = 5 for this example)
model <- knn(train = train_set[, selected_features], test = test_set[selected_features, ], cl = train_set$Class, k = 5)

# The predictions are directly obtained from the model
predictions <- as.numeric(as.character(model))

# Evaluate the model performance
cm_knn <- confusionMatrix(table(predictions, test_set$Class))

print(cm_knn)

# Print the accuracy
accuracy <- cm_knn$overall['Accuracy']
print(accuracy)
# Load necessary libraries
library(class)
library(caret)
library(keras)

# Preprocessing steps
data("BreastCancer")
BreastCancer <- na.omit(BreastCancer)
BreastCancer$Class <- ifelse(BreastCancer$Class == "benign", 0, 1)

# Splitting the data
set.seed(123)
trainIndex <- createDataPartition(BreastCancer$Class, p = 0.7, list = FALSE)
train_set <- BreastCancer[trainIndex, ]
test_set <- BreastCancer[-trainIndex, ]

# Preparing data for Keras model
train_x <- as.matrix(train_set[, -ncol(train_set)])
train_y <- train_set[, ncol(train_set)]
test_x <- as.matrix(test_set[, -ncol(test_set)])
test_y <- test_set[, ncol(test_set)]

# Define the Keras model
model_keras <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = c(ncol(train_x))) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
model_keras %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model
history <- model_keras %>% fit(
  train_x,
  train_y,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)

# Evaluate the model
keras_results <- model_keras %>% evaluate(test_x, test_y)
keras_accuracy <- keras_results[['accuracy']]
print(paste("Keras Model Accuracy:", keras_accuracy))


