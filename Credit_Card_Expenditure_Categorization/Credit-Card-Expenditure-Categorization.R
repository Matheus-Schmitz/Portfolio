#----------------------------------------------------------------#
## Directory
setwd("C:/DSA/MachineLearning/Cap11/1-Cap11-R/R")
getwd()

#----------------------------------------------------------------#
## Packages
library(dplyr)
library(caret)
library(gains)
library(pROC)
library(ROCR)
library(ROSE)
library(e1071)
library(mice)

#----------------------------------------------------------------#
## Data Loading
dataset_clients <- read.csv("dados/cartoes_clientes.csv")
View(dataset_clients)

#----------------------------------------------------------------#
## Exploratory Analysis
str(dataset_clients)
summary(dataset_clients)
summary(dataset_clients$card2spent)

# Removing the ID variables
dataset_clients <- dataset_clients[-1]

# Checking missing values
sapply(dataset_clients, function(x)sum(is.na(x)))

# Check if target variable is balanced
table(dataset_clients$Customer_cat)
prop.table(table(dataset_clients$Customer_cat))

## Visual Analysis

# Boxplot and Histogram
boxplot(dataset_clients$card2spent)
summary(dataset_clients$card2spent)
hist(dataset_clients$card2spent)

boxplot(dataset_clients$hourstv)
summary(dataset_clients$hourstv)
hist(dataset_clients$hourstv)

# scatter Plot
plot(dataset_clients@card2spent, dataset_clients$hourstv, xlab = "Card Spenditure", ylab = "TV Hours")

#----------------------------------------------------------------#
## Pre-processing data

# Function to factorize categorical variables
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(paste(df[[variable]]))
  }
  return(df)
}

# List of categorical variables
categorical.vars <- c('townsize', 'gender', 'jobcat', 'retire', 'hometype', 'addresscat', 
                      'cartype', 'carbought', 'card2', 'card2type', 
                      'card2benefit', 'bfast', 'internet', 'Customer_cat')

# Factorizing categorical variables
dataset_clients <- to.factors(df = dataset_clients, variables = categorical.vars)
str(dataset_clients)

## Applying imputation in missing values with the PMM (Predictive Mean Matching) method

# Checking missing values
sapply(dataset_clients, function(x)sum(is.na(x)))

# Retrieving the column number for the factor variables, to exclude them from imputation
fac_col <- as.integer(0)
facnames <- names(Filter(is.factor, dataset_clients))
k = 1

for (i in facnames){
  while (k <= 16){
    grep(i, colnames(dataset_clients))
    fac_col[k] <- grep(i, colnames(dataset_clients))
    k = k + 1
    break
  }
}

# Factor column
fac_col
str(dataset_clients[,c(fac_col)])

# Slicing dataset
View(dataset_clients)
View(dataset_clients[,c(fac_col)])

# Defining imputation rules
imputation_rules <- mice((dataset_clients[,-c(fac_col)]),
                         m = 1,
                         maxit = 50,
                         meth = 'pmm')

# Applying imputation rules
?mice::complete
total_data <- complete(imputation_rules, 1)
View(total_data)

# Rejoining the categorical variavles
dataset_clients_final <- cbind(total_data, dataset_clients[,c(fac_col)])
View(dataset_clients_final)

# Dimension
dim(dataset_clients_final)

# Data types
str(dataset_clients_final)

# Checking missing values
sapply(dataset_clients_final, function(x)sum(is.na(x)))
sum(is.na(dataset_clients_final))
sum(is.na(dataset_clients))

#----------------------------------------------------------------#
## Splitting dataset in 80% training and 20% testing

# Creating index to split rows
row_split_index <- sample(x = nrow(dataset_clients_final),
                           size = 0.8 * nrow(dataset_clients_final),
                           replace = FALSE)

# Splitting rows
data_train <- dataset_clients_final[row_split_index,]
data_test <- dataset_clients_final[-row_split_index,]

View(data_train)
View(data_test)

#----------------------------------------------------------------#
## Balancing the train dataset, since it's skewed

# Checking target variable balance
prop.table(table(data_train$Customer_cat))

## Using SMOTE to balance classes (SMOTE = Synthetic Minority Oversampling Technique)

# Package
library("DMwR")

# SMOTE
data_train_balanced <- SMOTE(Customer_cat ~ ., data_train, perc.over = 3000, perc.under = 200)

# Checking target variable balance
prop.table(table(data_train_balanced$Customer_cat))

# Checking data dimensions
dim(data_train_balanced)
dim(data_test)

#----------------------------------------------------------------#
## Saving and loading data

# Saving the finalized datasets
write.csv(data_train_balanced, "data_train_balanced.csv")
write.csv(data_test, "data_test.csv")

library(readr)
# Loading datasets
data_train_02 <- read_csv("data_train_balanced.csv")
data_test_02 <- read_csv("data_test.csv")

# Removing the X1 column created by the random indexing
data_train_02 <- data_train_02[!names(data_train_02) %in% c("X1")]
data_test_02 <- data_test_02[!names(data_test_02) %in% c("X1")]

# Might also need to re-change the data type for categoriacal variables

#----------------------------------------------------------------#
## Converting the target variable to numeric

# Checking the target variable
str(data_train_balanced$Customer_cat)

# Enconding variable (changing from string to numbers)
data_train_balanced$Customer_cat <- as.numeric(as.factor(data_train_balanced$Customer_cat))
data_test$Customer_cat <- as.numeric(as.factor(data_test$Customer_cat))
str(data_train_balanced$Customer_cat)
str(data_test$Customer_cat)

# Changing target variable type back to factor
data_train_balanced$Customer_cat <- as.factor(data_train_balanced$Customer_cat)
data_test$Customer_cat <- as.factor(data_test$Customer_cat)
str(data_train_balanced$Customer_cat)
str(data_test$Customer_cat)

#----------------------------------------------------------------#
## Predictive model v1

# Version 01 - Radial Kernel (RBF)
model_v1 <- svm(Customer_cat ~ ., data = data_train_balanced, na.action = na.omit, scale = TRUE)
summary(model_v1)
print(model_v1)

# Making predictions
predictions_v1 <- predict(model_v1, newdata = data_test)

# Confusion Matrix
results_v1 <- caret::confusionMatrix(predictions_v1, data_test$Customer_cat)
results_v1

# Metrics
library(multiROC)
roc_curve <- multiclass.roc(response = data_test$Customer_cat, predictor = predictions_v1)
class(data_test$Customer_cat)
class(predictions_v1)
v1_roc_curve <- multiclass.roc(response = data_test$Customer_cat, predictor = as.numeric(as.factor(predictions_v1)))

# AUC (area under curve) score
v1_roc_curve$auc

#----------------------------------------------------------------#
## Predictive model v2

# Version 02 - Linear Kernal & GridSearch on "cost" parameter
?tune

# Searching for best paramenters
v2_model_grid <- tune(svm,
                   Customer_cat ~ .,
                   data = data_train_balanced,
                   kernel = 'linear',
                   ranges = list(cost = c(0.05, 0.1, 0.5, 1, 2)))

summary(v2_model_grid)

# Best model parameters
v2_model_grid$best.parameters

# Creating best model
v2_model_grid$best.model
model_v2 <- v2_model_grid$best.model
summary(model_v2)

# Predictions
predictions_v2 <- predict(model_v2, data_test)

# Confusion Matrix
results_v2 <- caret::confusionMatrix(predictions_v2, data_test$Customer_cat)
results_v2

# AUC (area under curve) score
v2_roc_curve <- multiclass.roc(response = data_test$Customer_cat, predictor = as.numeric(as.factor(predictions_v2)))
v2_roc_curve$auc

#----------------------------------------------------------------#
## Predictive model v3

# Version 03 - Radial Kernel (RBF) & GridSearch on "cost" and "gamma" parameters

# Searching for best paramenters
v3_model_grid <- tune(svm,
                      Customer_cat ~ .,
                      data = data_train_balanced,
                      kernel = 'radial',
                      ranges = list(cost = c(0.05, 0.1, 0.5, 1, 2),
                                    gamma = c(0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2)))

summary(v3_model_grid)

# Best model parameters
v3_model_grid$best.parameters

# Creating best model
v3_model_grid$best.model
model_v3 <- v3_model_grid$best.model
summary(model_v3)

# Predictions
predictions_v3 <- predict(model_v3, data_test)

# Confusion Matrix
results_v3 <- caret::confusionMatrix(predictions_v3, data_test$Customer_cat)
results_v3

# AUC (area under curve) score
v3_roc_curve <- multiclass.roc(response = data_test$Customer_cat, predictor = as.numeric(as.factor(predictions_v3)))
v3_roc_curve$auc

#----------------------------------------------------------------#
## Predictive model v4

# Version 04 - Polinomial Kernel (RBF) & GridSearch on "cost" and "degree" parameters

# Searching for best paramenters
v4_model_grid <- tune(svm,
                      Customer_cat ~ .,
                      data = data_train_balanced,
                      kernel = 'polynomial',
                      ranges = list(cost = c(0.5, 1, 2),
                                    degree = c(2, 3, 4)))

summary(v4_model_grid)

# Best model parameters
v4_model_grid$best.parameters

# Creating best model
v4_model_grid$best.model
model_v4 <- v4_model_grid$best.model
summary(model_v4)

# Predictions
predictions_v4 <- predict(model_v4, data_test)

# Confusion Matrix
results_v4 <- caret::confusionMatrix(predictions_v4, data_test$Customer_cat)
results_v4

# AUC (area under curve) score
v4_roc_curve <- multiclass.roc(response = data_test$Customer_cat, predictor = as.numeric(as.factor(predictions_v4)))
v4_roc_curve$auc

#----------------------------------------------------------------#
## Comparing models

# model_v1
score_v1 <- c("Model v1 RBF Kernel", round(results_v1$ouverall['Accuracy']), round(v1_roc_curve$auc, 4))

# model_v2
score_v2 <- c("Model v2 Linear Kernel", round(results_v2$ouverall['Accuracy']), round(v2_roc_curve$auc, 4))

# model_v3
score_v3 <- c("Model v3 RBF Tunning", round(results_v3$ouverall['Accuracy']), round(v3_roc_curve$auc, 4))

# model_v4
score_v4 <- c("Model v4 Polynomial Kernel", round(results_v4$ouverall['Accuracy']), round(v4_roc_curve$auc, 4))

# Dataframe with all results
model_comparison <- rbind(score_v1, score_v2, score_v3, score_v4)
rownames(model_comparison) <- c('1', '2', '3', '4')
colnames(model_comparison) <- c('Model', 'Acuracy', 'AUC')
model_comparison <- as.data.frame(model_comparison)
View(model_comparison)

## Plots 
library(ggplot2)

# Accuracy plot
ggplot(model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity")

# AUC plot
ggplot(model_comparison, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity")

#----------------------------------------------------------------#
## Predicting with chosen model

# Saving selected model
?saveRDS
saveRDSmodel_v2("modelos/model_v2.rds")

# Loading saved model
svm_model <- readRDS("modelos/model_v2.rds")
print(svm_model)

# Loading new client data
new_clients <- read.csv("dados/novos_clientes.csv", header = TRUE)
View(new_clients)
dim(new_clients)

# Predicting
predictions_new_clients <- predict(svm_model, new_clients)

# Predictions
new_clients_expenditure_predictions <- data.frame(as.numeric(as.factor(predictions_new_clients)))
colnames(new_clients_expenditure_predictions) <- ("Expenditure Predictions")

# Client ages
ages_new_clients <- data.frame(new_clients$age)
colnames(ages_new_clients) <- ("Ages")

# Final dataframe
final_results <- cbind(ages_new_clients, new_clients_expenditure_predictions)
View(final_results)

# Adjusting the prediction label back to string
library(plyr)
final_results$'Expenditure Predictions' <- mapvalues(final_results$'Expenditure Predictions',
                                                     from = c(1, 2, 3),
                                                     to = c("High", "Medium", "Low"))
View(final_results)

# Saving final results
write.csv(final_results, "dados/final_results.csv")
