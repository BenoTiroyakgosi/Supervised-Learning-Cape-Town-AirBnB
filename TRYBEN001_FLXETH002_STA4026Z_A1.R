

# ==================== Q1 PREPARATION (run once) =====================
#loading libraries
library(readr)
library(dplyr)
library(caret)      #for training the models
library(ggplot2)
library(pROC)  # for ROC plotting 
library(knitr)

#loading data and making reproducible 80/20 split for training and test data respectively
listings <- read.csv("listings.csv")  
set.seed(123)  


train_idx <- sample(nrow(listings), size=0.8*nrow(listings))
train_raw <- listings[train_idx, ]
test_raw  <- listings[-train_idx, ]

# selecting the median as classification threshold
threshold <- 183


# Creating binary target factor for training with caret 
train_raw$HighAvailability <- factor(ifelse(train_raw$availability_365 > threshold, 1, 0),
                                     levels = c("0", "1"),
                                     labels = c("Low", "High"))
test_raw$HighAvailability  <- factor(ifelse(test_raw$availability_365  > threshold, 1, 0),
                                     levels = c("0", "1"),
                                     labels = c("Low", "High"))

# distribution of availability_365
hist(train_raw$availability_365,col = "lightblue", xlab = " availability_365", main = "Distribution of availability_365 ",breaks = 30)
abline(v = threshold, col = "red", lwd = 2, lty = 2)  


# Class imbalance on training set (report this in your report)
cat("\nClass distribution on training set:\n")
print(prop.table(table(train_raw$HighAvailability)))
print(prop.table(table(test_raw$HighAvailability)))

# Remove the availability_365, other variables on availability, host information 
train <- train_raw %>% select(-c(id, host_id, host_since,neighbourhood_ward, host_verifications,  availability_30,availability_60 ,availability_90, availability_365))
test  <- test_raw  %>% select(-c(id, host_id, host_since,neighbourhood_ward, host_verifications,  availability_30,availability_60 ,availability_90, availability_365))




# =================================================================

# ====================== COMMON CV SETUP ==========================
trControl <- trainControl(method = "cv",
                          number = 10,
                          classProbs = TRUE,
                          summaryFunction = twoClassSummary,
                          savePredictions = TRUE,
                          verboseIter = FALSE)

# =================================================================
# ==================== 1. REGULARISED ELASTIC-NET LOGISTIC REGRESSION
# =================================================================

set.seed(321)

# defining hyperparameter search space 

tuneGrid_en <- expand.grid(alpha = seq(0, 1, by = 0.1),
                           lambda = 10^seq(-4, 1, length.out = 20))

# training elastic net logistic regression and performing cross-validation
model_en <- train(HighAvailability ~ .,
                  data = train,
                  method = "glmnet",
                  family = "binomial",
                  trControl = trControl,
                  tuneGrid = tuneGrid_en,
                  metric = "ROC")

?train

#save model to working directory
saveRDS(model_en, file = "final_elasticnet_model.rds")


#loading model from working directory
model_en <- readRDS("final_elasticnet_model.rds")

coef(model_en$finalModel, s = model_en$bestTune$lambda)
# Best hyperparameters 
print(model_en$bestTune)

#plotting CV results

#extracting lambda from best performing model
best_lambda <- log10(model_en$bestTune$lambda)


# the best model is with alpha = 0 but model with alpha =  0.5 is selected
#extracting results of best model with alpha = 0.5 and lambda =  0.001129 
alpha_05_results<- model_en$results%>%
  filter(alpha == 0.5)

best_lambda_for_alpha_05 <-round(alpha_05_results$lambda[which.max(alpha_05_results$ROC)],6)



ggplot(model_en$results, aes(x = log10(lambda), y = ROC, color = factor(alpha))) +
  geom_line() +
  geom_point() +
  labs(
    x = "log10(lambda)",
    y = "Cross-validated ROC (AUC)",
    color = "Alpha",
    title = ""
  ) +
  theme_minimal()+
  geom_vline(xintercept = best_lambda, linetype = "dashed", color = "red")+
  geom_vline(xintercept = log10(best_lambda_for_alpha_05), linetype = "dashed", color = "blue")+
  scale_y_continuous(limits = c(0.55,0.62))



# getting predictions for best model at alpha = 0.5
en_pred <- model_en$pred%>%
  filter(alpha == 0.5)
en_results_05 <- model_en$results%>%
  filter(alpha == 0.5)

# creating a rounded version of lambda to allow for filtering 
en_pred$round_lambda <- round(en_pred$lambda, 6)

#filtering  predictions corresponding to lambda 
en_pred_final <- en_pred%>%
  filter(round_lambda ==  best_lambda_for_alpha_05)


# ----------------- Extract CV metrics at τ = 0.5 -----------------
cv_pred_en <- en_pred_final
cv_pred_en$pred_class <- ifelse(cv_pred_en$High > 0.5, "High", "Low")

cm_en <- confusionMatrix(factor(cv_pred_en$pred_class, levels = c("Low", "High")),
                         factor(cv_pred_en$obs, levels = c("Low", "High")),
                         positive = "High")

# Metrics for table (rounded to 3 decimals in report)
acc_en    <- round(cm_en$overall["Accuracy"], 3)
prec_en   <- round(cm_en$byClass["Pos Pred Value"], 3)
rec_en    <- round(cm_en$byClass["Sensitivity"], 3)
spec_en   <- round(cm_en$byClass["Specificity"], 3)
f1_en     <- round(2 * prec_en * rec_en / (prec_en + rec_en), 3)
auroc_en  <- round(en_results_05$ROC[which.max(en_results_05$ROC)], 3)  # CV AUROC of best model

cat("\nElastic-net CV metrics (τ=0.5):\n")
print(c(Accuracy = acc_en, F1 = f1_en, Precision = prec_en, 
        Recall = rec_en, Specificity = spec_en, AUROC = auroc_en))

# =================================================================
# ==================== 2. RANDOM FOREST
# =================================================================
library(doParallel)
library(ranger)   

# 1. Start parallel backend for CV folds
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(4026)

#setting up cv instructions
trControl <- trainControl(method = "cv",
                          number = 10,                   # number of folds (k)
                          classProbs = TRUE,             # produce prediction probabilities
                          summaryFunction = twoClassSummary, # returns ROC(AUC), recall and specificity
                          savePredictions = TRUE,
                          allowParallel = TRUE,     
                          verboseIter = FALSE) # controls printing during training 

# 2. Ranger-specific tuning + internal threads
tuneGrid_ranger <- expand.grid(
  mtry = 2:(ncol(train) - 1),
  splitrule = "gini",           
  min.node.size = c(5, 10, 20)
)

# training model and performing cv

model_rf <- train(HighAvailability ~ .,
                  data = train,
                  method = "ranger",          
                  trControl = trControl,
                  tuneGrid = tuneGrid_ranger,
                  metric = "ROC",
                  num.trees = 500,
                  num.threads = 1,   # ← ranger's own threads
                  verbose = FALSE)
            

# stopping pcluster used in parallel computing
stopCluster(cl)

#saving  and readind file 
saveRDS(model_rf, file = "Final_Random_Forest.rds")

model_rf <- readRDS("Final_Random_Forest.rds")

# Best model has number of random features selected as 18 and node size 5
print(model_rf$bestTune)

#plotting cv 
plot(model_rf)

#selected model with which randomly select 9 features and has min node size of 10
results_rf <- model_rf$results
  

node_20 <- results_rf %>%
  filter(min.node.size == 20)

#selecting cv prediction for model with min node = 20 and number of features = 4

pred_rf <-model_rf$pred%>%
  filter(min.node.size == 20, mtry == 4)


# ----------------- Extract CV metrics at τ = 0.5 -----------------
cv_pred_rf <- pred_rf
cv_pred_rf$pred_class <- ifelse(cv_pred_rf$High > 0.5, "High", "Low")

cm_rf <- confusionMatrix(factor(cv_pred_rf$pred_class, levels = c("Low", "High")),
                         factor(cv_pred_rf$obs, levels = c("Low", "High")),
                         positive = "High")

acc_rf    <- round(cm_rf$overall["Accuracy"], 5)
prec_rf   <- round(cm_rf$byClass["Pos Pred Value"], 5)
rec_rf    <- round(cm_rf$byClass["Sensitivity"], 5)
spec_rf   <- round(cm_rf$byClass["Specificity"], 5)
f1_rf     <- round(2 * prec_rf * rec_rf / (prec_rf + rec_rf), 5)
auroc_rf  <- round(node_20$ROC[which.max(node_20$ROC)], 3)

cat("\nRandom Forest CV metrics (τ=0.5):\n")
print(c(Accuracy = acc_rf, F1 = f1_rf, Precision = prec_rf, 
        Recall = rec_rf, Specificity = spec_rf, AUROC = auroc_rf))

# ==================== QUESTION 3(a): CV F1 vs threshold ====================

# Use CV predictions from the selected random forest model
cv_pred_q3 <- pred_rf


# sequence of tau values for predictions
tau_grid <- seq(0, 1, by = 0.001)

# empty vector for storing f1_values
f1_vals <- numeric(length(tau_grid))

# computing f1 values given tau
for (i in seq_along(tau_grid)) {
  
  tau <- tau_grid[i]
  
  pred_class <- ifelse(cv_pred_q3$High > tau, "High", "Low")
  
  cm <- confusionMatrix(
    factor(pred_class, levels = c("Low", "High")),
    factor(cv_pred_q3$obs, levels = c("Low", "High")),
    positive = "High"
  )
  
  precision <- cm$byClass["Pos Pred Value"]
  recall    <- cm$byClass["Sensitivity"]
  
  if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
    f1_vals[i] <- NA # controlling for computation errors that may result from very high or low values of tau 
  } else {
    f1_vals[i] <- 2 * precision * recall / (precision + recall)
  }
}

#extracting best tau and f1 to for point on plot 
best_tau <- tau_grid[which.max(f1_vals)]
best_f1  <- max(f1_vals, na.rm = TRUE)

#plotting cv f1 scores
plot(tau_grid, f1_vals, type = "l", lwd = 2,
     xlab = expression(tau),
     ylab = "Cross-validated F1 score",
     main = "CV F1 score as a function of threshold")

abline(v = best_tau, col = "red", lty = 2, lwd = 2)
points(best_tau, best_f1, pch = 19)

cat("Best threshold:", best_tau, "\n")
cat("Maximum CV F1:", round(best_f1, 3), "\n")


# ==================== QUESTION 3(b): CV ROC====================


#ROC object storing real observations and their classification probabilities 
roc_rf <- roc(
  response = pred_rf$obs,
  predictor = pred_rf$High,
  levels = c("Low", "High")
)

# Extract all ROC coordinates (Values of Specificity and sensitivity at all thresholds)
roc_coords <- coords(
  roc_rf,
  x = roc_rf$thresholds,
  input = "threshold",
  ret = c("specificity", "sensitivity"),
  transpose = FALSE
)

# Create FPR and TPR for plotting 
fpr <- 1 - roc_coords[, "specificity"]
tpr <- roc_coords[, "sensitivity"]

# Plot standard ROC: TPR vs FPR
plot(fpr, tpr, type = "l", lwd = 2,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC curve (Random Forest, CV)")

abline(0, 1, lty = 2, col = "grey")

# Coordinates for point of best threshold
coords_best <- coords(
  roc_rf,
  x = best_tau,
  input = "threshold",
  ret = c("specificity", "sensitivity")
)

#point of best threshold 
best_fpr <- 1 - coords_best["specificity"]
best_tpr <- coords_best["sensitivity"]

points(best_fpr, best_tpr, pch = 19, col = "red", cex = 1.3)

text(best_fpr, best_tpr,
     labels = paste0("\n Threshold = ",round(best_tau, 2),
                     "\n F1 = ", round(best_f1, 3)),
     pos = 1)


################################  3 (C)


#fitting selected model on training data 


set.seed(4026)

rf_final_q3 <- train(
  HighAvailability ~ .,
  data = train,
  method = "ranger",
  importance = T,
  trControl = trainControl(
    method = "none",
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = data.frame(
    mtry = 4,
    splitrule = "gini",
    min.node.size = 20
  ),
  metric = "ROC",
  num.trees = 500,
  num.threads = 1,
  verbose = FALSE
)
# Question 3 model training for predition with test set 

rf_final_q3 <- train(
  HighAvailability ~ .,
  data = train,
  method = "ranger",
  importance = "permutation",   # ← important fix
  trControl = trainControl(
    method = "none",
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = data.frame(
    mtry = 4,
    splitrule = "gini",
    min.node.size = 20
  ),
  metric = "ROC",
  num.trees = 500,
  num.threads = 1,
  verbose = FALSE
)



saveRDS(rf_final_q3, file = "RF_trained_for_Q4.rds")


rf_final_q3 <- readRDS("RF_trained_for_Q4.rds")
test_prob_rf <- predict(rf_final_q3, newdata = test, type = "prob")[, "High"]


# Classify using optimal threshold from Question 3(a)
test_pred_rf <- ifelse(test_prob_rf > best_tau, "High", "Low")

# Confusion matrix
cm_test_rf <- confusionMatrix(
  factor(test_pred_rf, levels = c("Low", "High")),
  factor(test$HighAvailability, levels = c("Low", "High")),
  positive = "High"
)

# Test metrics
acc_test_rf  <- round(cm_test_rf$overall["Accuracy"], 3)
prec_test_rf <- round(cm_test_rf$byClass["Pos Pred Value"], 3)
rec_test_rf  <- round(cm_test_rf$byClass["Sensitivity"], 3)
spec_test_rf <- round(cm_test_rf$byClass["Specificity"], 3)
f1_test_rf   <- round(2 * prec_test_rf * rec_test_rf / (prec_test_rf + rec_test_rf), 3)

roc_test_rf <- roc(
  response = test$HighAvailability,
  predictor = test_prob_rf,
  levels = c("Low", "High")
)
auroc_test_rf <- round(as.numeric(auc(roc_test_rf)), 3)

# CV metrics from Question 3(c) at best_tau
acc_cv_rf   <- acc_rf
prec_cv_rf  <- prec_rf
rec_cv_rf   <- rec_rf
spec_cv_rf  <- spec_rf
f1_cv_rf    <- f1_rf
auroc_cv_rf <- auroc_rf

# Comparison table
q3c_table <- data.frame(
  Data = c("Cross-validation", "Test"),
  Accuracy = c(acc_cv_rf, acc_test_rf),
  F1 = c(f1_cv_rf, f1_test_rf),
  Precision = c(prec_cv_rf, prec_test_rf),
  Recall = c(rec_cv_rf, rec_test_rf),
  Specificity = c(spec_cv_rf, spec_test_rf),
  AUROC = c(auroc_cv_rf, auroc_test_rf)
)

kable(q3c_table, digits = 3,
      caption = "Random forest performance at the optimal threshold")

## Question 4c (Misclassification )

test_pred_results <- cbind(test,test_prob_rf  ,test_pred_rf)
#random sample of false positive and false negative observations
set.seed(123)

fp_all <- test_pred_results[
  test_pred_results$HighAvailability == "Low" &
    test_pred_results$test_pred_rf == "High",
]

false_positives <- fp_all[
  sample(nrow(fp_all), 5),
]


fn_all <- test_pred_results[
  test_pred_results$HighAvailability == "High" &
    test_pred_results$test_pred_rf == "Low",
]

false_negatives <- fn_all[
  sample(nrow(fn_all), 5),
]

fp_show <- false_positives[,  c("maximum_nights_avg_ntm", "host_listings_count",
                                "instant_bookable", "host_acceptance_rate", "latitude",
                                "HighAvailability", "test_pred_rf")]

fn_show <- false_negatives[, c("maximum_nights_avg_ntm", "host_listings_count",
                               "instant_bookable", "host_acceptance_rate", "latitude",
                               "HighAvailability", "test_pred_rf")]

colnames(fp_show) <- c(
  "Max nights",
  "Host count",
  "Instant",
  "Accept_rate",
  "Lat",
  "True",
  "Pred"
)


colnames(fn_show) <- c(
  "Max nights",
  "Host count",
  "Instant",
  "Accept_rate",
  "Lat",
  "True",
  "Pred"
)


knitr::kable(fp_show, caption = "Five false positives from the test set")
knitr::kable(fn_show, caption = "Five false negatives from the test set")
