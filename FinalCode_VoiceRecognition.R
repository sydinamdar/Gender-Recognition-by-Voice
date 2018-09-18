######## PCA----------=======================================================
library(leaps)
library(MASS)
library(klaR)
library(e1071)
library(ROCR)
library(tree)
library(caret)
library(glmnet)
library(InformationValue)
library(ggplot2)




#reading the data and splitting into features and label
set.seed(28)

#Real time load file
mp3_input_path="C:\\Users\\syd24\\Downloads\\"

## File Path
filePath=read.csv("C:\\Users\\syd24\\OneDrive\\Desktop\\WPI\\Statistics\\Final Project\\voice.csv", sep = ",")
voice <- data.frame(filePath)


voice <- data.frame(filePath)
voice_input <- voice[,-21]
voice_label <- data.frame(voice[,21])
names(voice_label) <- c("Gender")


#normalizing data
voice_input_normalised <- voice_input


# hot encoding
# repeat 9 for the entire length of examples
lab <- rep(9,as.integer(dim(voice_label)[1]))

# 1 - male
# 0 - female
for (i in c(1:dim(voice_label)[1])){
  if(voice_label[i,1] == "male"){
    lab[i] = 1
  } else{
    lab[i] = 0
  }
}


#combine for shuffle
#voice_final <- data.frame(cbind(voice_input_normalised, voice_label,lab))
voice_final <- data.frame(cbind(voice_input, voice_label,lab))

#shuffle
voice_final <- voice_final[sample(nrow(voice_final),nrow(voice_final)),]



#drop label
voice_final.lab <- voice_final[,22]
voice_final <- voice_final[,-21:-22]



#Working data
# work_data <- head(voice_final, b-a)
# work_data.lab <- head(voice_final.lab,b-a)
work_data =voice_final
work_data.lab=voice_final.lab



#Confusion matrix function
confusion_matrix_plot = function(a,c){
  TClass <- factor(c("Male", "Female", "Male", "Female"))
  PClass <- factor(c("Male", "Male", "Female", "Female"))
  levels(PClass) = c("Male","Female")
  Y      <- c(a[3],a[1],a[4],a[2])
  df <- data.frame(TClass, PClass, Y)
  
  
  ggplot(data =  df, mapping = aes(x = TClass, y = PClass)) +
    geom_tile(aes(fill = Y), colour = "white") +
    geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1, colour = "darkblue", alpha = 0.8, size = 12) +
    scale_fill_gradient(low = "brown1", high = "chartreuse3") +
    theme_bw() + theme(legend.position = "none") +
    ggtitle(c)
  
  
}


#ROC function
roc_plot = function(actual,predicted){
  InformationValue::plotROC(actual, predicted)
  cat("AUC = ",InformationValue::AUROC(actual, predicted),"\n")
}


#Gain and Lift chart
gain_lift_plot = function(pred){
  par(mfrow=c(1,2))
  perf_lift <- performance(pred,"lift","rpp")
  perf_gain <- performance(pred,"tpr","rpp")
  plot(perf_lift,main="Lift Curve",col="red",lwd =2)
  plot(perf_gain,main="Gain Curve",col="orange",lwd=2)
}


#Kolmogorov-Smirnov plot
ks_chart = function(actual,predicted){
  ks <- ks_stat(actual, predicted)
  cat("KS-Statistic",ks,"\n")
  print(ks_stat(actual,predicted,returnKSTable = T))
  ks_plot(actual,predicted)
}



#pca
pca =  prcomp(work_data,center = TRUE,scale = TRUE)
pc.comp <- pca$scores
biplot(pca,scale=0)
pca.var = pca$sdev^2
pca.var
pve = pca.var/sum(pca.var)
pve
plot(pve,xlab = "Principal Components",ylab = "PVE",ylim=c(0,1),type = 'b')

#selected first two principal components at knee
pca.data <- pca$x[,1:2]
#selected three principal components
pca3.data <- pca$x[,1:3]
#selected four principal components
pca4.data <- pca$x[,1:4]
#selected five principal components
pca5.data <- pca$x[,1:5]
#select pc1,pc2,pc4,PC5
pcacomb.data <- pca$x[,c(1,2,4,5)]


pca.data <- data.frame(cbind(pca.data,work_data.lab))
pca3.data <- data.frame(cbind(pca3.data,work_data.lab))
pca4.data <- data.frame(cbind(pca4.data,work_data.lab))
pca5.data <- data.frame(cbind(pca5.data,work_data.lab))
pcacomb.data <- data.frame(cbind(pcacomb.data,work_data.lab))


names(pca.data) <- c("PC1","PC2","lab")
names(pca3.data) <- c("PC1","PC2","PC3","lab")
names(pca4.data) <- c("PC1","PC2","PC3","PC4","lab")
names(pca5.data) <- c("PC1","PC2","PC3","PC4","PC5","lab")
names(pcacomb.data) <- c("PC1","PC2","PC4","PC5","lab")


###Safe
a <- 200
b <- dim(pca.data)[1]

#Put away this data in the safe.
safe_data_pca <- tail(pca.data,a)
#work data
pca.data=head(pca.data,b-a)

#Put away this data in the safe.
safe_data_pca3 <- tail(pca3.data,a)
#work data
pca3.data=head(pca3.data,b-a)

#Put away this data in the safe.
safe_data_pca4 <- tail(pca4.data,a)
#work data
pca4.data=head(pca4.data,b-a)


#Put away this data in the safe.
safe_data_pca5 <- tail(pca5.data,a)
#work data
pca5.data=head(pca5.data,b-a)


#Put away this data in the safe.
safe_data_pcomb <- tail(pcacomb.data,a)
#work data
pcacomb.data=head(pcacomb.data,b-a)




#building model
#split data into validation and training
train <- rep(c(TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE), as.integer(dim(pca.data)/12)[1])
test <- !train
num <- as.integer(dim(pca.data)[1]*0.8)
train <- sample(dim(pca.data)[1], num)


#Validation data with 2 pcs
validation_data <- pca.data[-train,]
#validation data with 3 pcs
validation_data3 <- pca3.data[-train,]
#validation data with 4 pcs
validation_data4 <- pca4.data[-train,]
#validation data with 5 pcs
validation_data5 <- pca5.data[-train,]
#validation data with combination of pcs
validation_datacomb <- pcacomb.data[-train,]

#Training data with 2pcs
training_data <- pca.data[train,]
#training data with 3pcs
training_data3 <- pca3.data[train,]
#training data with 4pcs
training_data4 <- pca4.data[train,]
#training data with 5pcs
training_data5 <- pca5.data[train,]
#training data with combination of pcs
training_datacomb <- pcacomb.data[train,]


#######################LOGISTIC REGRESSION###########################################

#k fold cross-validation for logistic regression 
kfold_cv = function(pca.data){
  #fpr <- NULL
  #fnr <- NULL
  
  #number of iterations
  k <- 500 #check later
  
  
  #accuracy
  acc <- NULL
  
  for (i in 1:k) {
    #train-test splitting
    sample_size <- floor(0.95*nrow(pca.data))
    index <- sample(seq_len(nrow(pca.data)),size = sample_size)
    cv.train <- pca.data[index,]
    cv.validate <- pca.data[-index,]
    
    cv.logistic_model <- glm(lab~.,data = cv.train,family = "binomial")
    
    cv.predict <- predict(cv.logistic_model,cv.validate,type = "response")
    
    cv.predition <- rep(1,dim(cv.validate)[1])
    cv.predition[cv.predict<0.5] <- 0
    correct_prediction <- mean(cv.predition==cv.validate$lab)
    table(cv.predition,cv.validate$lab)
    
    acc[i] <- correct_prediction
    
  }
  
  
  
  
  mean_acc <- mean(acc)
  mean_acc
  
  par(mfcol=c(1,2))
  
  #histogram for accuracy
  hist(acc,xlab = "Accuracy",ylab = "Frequency",col = "cyan",border = "blue",density = 30)
  
  #boxplot of accuracy
  boxplot(acc,col = 'cyan',border = 'blue',horizontal = T,xlab='Accuracy',main='Accuracy CV')
  return(mean_acc)
}

kfold_cv(pca.data)
kfold_cv(pca3.data)
kfold_cv(pca4.data)
kfold_cv(pca5.data)
kfold_cv(pcacomb.data)


#Logistic regression - ROC Curve, KS -plot, Lift chart,choose the best PC data
logistic_model <- glm(lab~.,data=training_datacomb,family = "binomial")
#prediction of validation
predict_logit <- predict(logistic_model,validation_datacomb,type = "response")


model.prediction <- rep(1,dim(validation_datacomb)[1])
model.prediction[predict_logit<0.5] <- 0
pred <- prediction(model.prediction,validation_datacomb$lab)
#Confusion matrix
confusion_matrix_plot(table(model.prediction,validation_datacomb$lab),"Logistic Regression")
#ROC 
roc_plot(validation_datacomb$lab,model.prediction)
#Gain and Lift chart
gain_lift_plot(pred)
#KS Statistic
ks_chart(validation_datacomb$lab,model.prediction)



#######################LOGISTIC REGRESSION###########################################



####################### LDA ####################################################
#k fold cross-validation for LDA
kfold_cv_lda = function(pca.data){
  fpr <- NULL
  fnr <- NULL
  
  #number of iterations
  k <- 500 #check later
  
  
  #accuracy
  acc <- NULL
  
  for (i in 1:k) {
    #train-test splitting
    sample_size <- floor(0.95*nrow(pca.data))
    index <- sample(seq_len(nrow(pca.data)),size = sample_size)
    cv.train <- pca.data[index,]
    cv.validate <- pca.data[-index,]
    
    cv.lda_model <- lda(lab~.,data = cv.train)
    
    cv.predict <- predict(cv.lda_model,cv.validate)
    
    correct_prediction <- mean(cv.predict$class==cv.validate$lab)
    #table(cv.predition,cv.validate$lab)
    #confusion matrix?
    
    acc[i] <- correct_prediction
    
  }
  
  
  mean_acc <- mean(acc)
  
  par(mfcol=c(1,2))
  
  #histogram for accuracy
  hist(acc,xlab = "Accuracy",ylab = "Frequency",col = "cyan",border = "blue",density = 30)
  
  #boxplot of accuracy
  boxplot(acc,col = 'cyan',border = 'blue',horizontal = T,xlab='Accuracy',main='Accuracy CV')
  return(mean_acc)
}

kfold_cv_lda(pca.data)
kfold_cv_lda(pca3.data)
kfold_cv_lda(pca4.data)
kfold_cv_lda(pca5.data)
kfold_cv_lda(pcacomb.data)

#LDA  - ROC Curve, KS- Chart , Gain and Lift chart(choose the best PC data)
#roc_curve_lda = function(training_data,validation_data){
lda_model <- lda(lab~.,data = training_datacomb)
predict_lda <- predict(lda_model,validation_datacomb)
pred.lda <- predict_lda$posterior[,2]
pred <- prediction(pred.lda,validation_datacomb$lab)
#Confusion matrix
confusion_matrix_plot(table(predict_lda$class,validation_datacomb$lab),"LDA")
#ROC 
roc_plot(validation_datacomb$lab,pred.lda)
#Gain and Lift chart
gain_lift_plot(pred)
#KS Statistic
ks_chart(validation_datacomb$lab,pred.lda)


####################### LDA ####################################################





###################### QDA #####################################################

#k fold cross-validation for QDA
kfold_cv_qda = function(pca.data){
  fpr <- NULL
  fnr <- NULL
  
  #number of iterations
  k <- 500 #check later
  
  
  #accuracy
  acc <- NULL
  
  for (i in 1:k) {
    #train-test splitting
    sample_size <- floor(0.95*nrow(pca.data))
    index <- sample(seq_len(nrow(pca.data)),size = sample_size)
    cv.train <- pca.data[index,]
    cv.validate <- pca.data[-index,]
    
    cv.qda_model <- qda(lab~.,data = cv.train)
    
    cv.predict <- predict(cv.qda_model,cv.validate)
    
    correct_prediction <- mean(cv.predict$class==cv.validate$lab)
    #table(cv.predition,cv.validate$lab)
    #confusion matrix?
    
    acc[i] <- correct_prediction
    
  }
  
  
  mean_acc <- mean(acc)
  
  par(mfcol=c(1,2))
  
  #histogram for accuracy
  hist(acc,xlab = "Accuracy",ylab = "Frequency",col = "cyan",border = "blue",density = 30)
  
  #boxplot of accuracy
  boxplot(acc,col = 'cyan',border = 'blue',horizontal = T,xlab='Accuracy',main='Accuracy CV')
  return(mean_acc)
}

kfold_cv_qda(pca.data)
kfold_cv_qda(pca3.data)
kfold_cv_qda(pca4.data)
kfold_cv_qda(pca5.data)
kfold_cv_qda(pcacomb.data)

#QDA  - ROC Curve, KS- Chart , Gain and Lift chart (choose the data for best PCs)
#roc_curve_qda = function(training_data,validation_data){
qda_model <- qda(lab~.,data = training_datacomb)
predict_qda <- predict(qda_model,validation_datacomb)
pred.qda <- predict_qda$posterior[,2]
pred <- prediction(pred.qda,validation_datacomb$lab)
#Confusion matrix
confusion_matrix_plot(table(predict_qda$class,validation_datacomb$lab),"QDA")
#ROC 
roc_plot(validation_datacomb$lab,pred.qda)
#Gain and Lift chart
gain_lift_plot(pred)
#KS Statistic
ks_chart(validation_datacomb$lab, pred.qda)


###################### QDA #####################################################



######################## SVM ###################################################
#svM with linear kernel
tune.out_linear = tune(svm,as.factor(lab)~.,data = training_datacomb,kernel="linear",ranges = list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100)),tunecontrol = tune.control(cross = 10))
summary(tune.out_linear)
svmfit = tune.out_linear$best.model
summary(svmfit)

svm.predict = predict(svmfit,validation_datacomb)
table(predict = svm.predict,truth=as.factor(validation_datacomb$lab))
acc_linear <- mean(svm.predict==validation_datacomb$lab)
acc_linear

#SVM with radial kernel
tune.out_radial=tune(svm , as.factor(lab)~., data=training_datacomb, kernel ="radial",ranges =list(cost=c(0.1 ,1 ,10 ,100 ,1000),gamma=c(0.5,1,2,3,4)),tunecontrol = tune.control(cross = 10))
svmfit_radial = tune.out_radial$best.model
summary(svmfit_radial)
svm.predict_radial = predict(svmfit_radial,validation_datacomb)
table(predict = svm.predict_radial,truth=as.factor(validation_datacomb$lab))
acc_radial <- mean(svm.predict_radial==validation_datacomb$lab)
acc_radial



#SVM  - ROC Curve, KS- Chart , Gain and Lift chart (choose data for best PCs)
#choose the best cost based on analysis
#linear svm
svm_model_linear <- svm(as.factor(lab)~.,data = training_datacomb,kernel="linear",cost=0.1)
predict_svm <- predict(svm_model_linear,validation_datacomb)
pred <- prediction(as.numeric(predict_svm),validation_datacomb$lab)
#Confusion matrix
confusion_matrix_plot(table(predict_svm,validation_datacomb$lab),"SVM")
#ROC 
roc_plot(validation_datacomb$lab,as.numeric(predict_svm))
#Gain and Lift chart
gain_lift_plot(pred)
#KS Statistic
ks_chart(validation_datacomb$lab, as.numeric(predict_svm))


#choose the best cost based on analysis
#radial svm
svm_model_radial <- svm(as.factor(lab)~.,data = training_datacomb,kernel="radial",cost=10,gamma=0.5)
predict_svm <- predict(svm_model_radial,validation_datacomb)
pred <- prediction(as.numeric(predict_svm),validation_datacomb$lab)
#Confusion matrix
confusion_matrix_plot(table(predict_svm,validation_datacomb$lab),"SVM")
#ROC
roc_plot(validation_datacomb$lab,as.numeric(predict_svm))
#Gain and Lift chart
gain_lift_plot(pred)  
#KS Statistic
ks_chart(validation_datacomb$lab, as.numeric(predict_svm))

######################## SVM ###################################################

logit_safe_predict <- predict(logistic_model,safe_data_pcomb,type = "response")
logit_safe_prediction <- rep(1,dim(safe_data_pcomb)[1])
logit_safe_prediction[logit_safe_predict<0.5] <- 0
logit_safe_accuracy <- mean(logit_safe_prediction==safe_data_pcomb$lab)
logit_safe_accuracy

lda_safe_predict <- predict(lda_model,safe_data_pcomb)
lda_safe_prediction <- mean(lda_safe_predict$class==safe_data_pcomb$lab)
lda_safe_prediction

qda_safe_predict <- predict(qda_model,safe_data_pcomb)
qda_safe_prediction <- mean(qda_safe_predict$class==safe_data_pcomb$lab)
qda_safe_prediction

svm_linear_safe_predict <- predict(svmfit,safe_data_pcomb)
svm_linear_safe_prediction <- mean(svm_linear_safe_predict==safe_data_pcomb$lab)
svm_linear_safe_prediction

svm_radial_safe_predict <- predict(svm_model_radial,safe_data_pcomb)
svm_radial_safe_prediction <- mean(svm_radial_safe_predict==safe_data_pcomb$lab)
svm_radial_safe_prediction



#######--------- PCA ends =========================================




rm(list=ls())

set.seed(12345)
library(leaps)
library(MASS)
library(bestglm)
library(glmnet)
library(e1071)
library(ggplot2)
library(ROCR)
library(caret)
library(gridExtra)
library(InformationValue)
library(tree)
library(randomForest)
library(gbm)
library(leaps)
library(tuneR)
library(mice)
library(klaR)



#Real time load file
mp3_input_path="C:\\Users\\syd24\\Downloads\\"

## File Path
filePath=read.csv("C:\\Users\\syd24\\OneDrive\\Desktop\\WPI\\Statistics\\Final Project\\voice.csv", sep = ",")
voice <- data.frame(filePath)


#process data
process_data <- function(raw_data){
  voice_input <- raw_data[,-21]
  voice_label <- data.frame(raw_data[,21])
  names(voice_label) <- c("Gender")
  
  
  mf_sd <- log(voice[,1]*voice[,2])
  quartile_a <- voice[,4] - 1.5 * voice[,6]
  quartile_b <- voice[,5] + 1.5 * voice[,6]
  quartile <- ((quartile_a)/(quartile_b)**2)
  skew_kurt <- log(voice[,7]* voice[,8])
  sp_ent_sfm <- voice[,9]*voice[,10]
  mode <- voice[,11]
  mean_fun <- log(voice[,13])
  min_max_fun <- log(voice[,14]*voice[,15])
  df_range <- voice[,19]
  
  #combine significant parameters
  voice_fe <- data.frame(cbind(mf_sd,quartile,skew_kurt,sp_ent_sfm,mode,mean_fun,min_max_fun,df_range))
  names(voice_fe) <- c("Mean function and Std dev","distribution", "Skew and Kurt", "Spectral entropy and sfm", "Mode", "Mean fun", "Min max fun","Df range")
  
  print(voice_fe)
  voice_fe_normalised <- voice_fe
  
  # #normalize data
  # for (i in names(voice_fe)) {
  #   voice_fe_normalised[i] <- (voice_fe[i]*sapply(voice_fe[i],mean))/sapply(voice_fe[i],sd)
  # }
  
  # hot encoding
  lab <- rep(9,as.integer(dim(voice_label)[1]))
  
  for (i in c(1:dim(voice_label)[1])){
    if(voice_label[i,1] == "male"){
      lab[i] = 1
    } else{
      lab[i] = 0
    }
  }
  
  #combine for shuffle
  voice_final <- data.frame(cbind(voice_fe_normalised, voice_label,lab))
  
  #drop label
  voice_final <- voice_final[,-9]
  
  #shuffle
  voice_final <- voice_final[sample(nrow(voice_final),nrow(voice_final)),]
  
  
  voice_final
}



#Confusion matrix

confusion_matrix_plot = function(a,c){
  TClass <- factor(c("Male", "Female", "Male", "Female"))
  PClass <- factor(c("Male", "Male", "Female", "Female"))
  levels(PClass) = c("Male","Female")
  Y      <- c(a[3],a[1],a[4],a[2])
  df <- data.frame(TClass, PClass, Y)
  
  
  ggplot(data =  df, mapping = aes(x = TClass, y = PClass)) +
    geom_tile(aes(fill = Y), colour = "white") +
    geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1, colour = "darkblue", alpha = 0.8, size = 12) +
    scale_fill_gradient(low = "brown1", high = "chartreuse3") +
    theme_bw() + theme(legend.position = "none") +
    ggtitle(c)
  
}


#process raw live data
process_live <- function(raw_data){
  voice <- raw_data
  #removed kurt
  mf_sd <- log(voice[,1]*voice[,2])
  quartile_a <- voice[,4] - 1.5 * voice[,6]
  quartile_b <- voice[,5] + 1.5 * voice[,6]
  quartile <- ((quartile_a)/(quartile_b)**2)
  skew_kurt <- log(voice[,7]* voice[,8])
  sp_ent_sfm <- voice[,9]*voice[,10]
  mode <- voice[,11]
  mean_fun <- log(voice[,13])
  min_max_fun <- log(voice[,14]*voice[,15])
  df_range <- voice[,19]
  
  #combine significant parameters
  voice_fe <- data.frame(cbind(mf_sd,quartile,skew_kurt,sp_ent_sfm,mode,mean_fun,min_max_fun,df_range))
  names(voice_fe) <- c("Mean.function.and.Std.dev","distribution", "Skew.and.Kurt", "Spectral.entropy.and.sfm", "Mode", "Mean.fun", "Min.max.fun","Df.range")
  
  print(voice_fe)
  
  voice_fe_normalised <- voice_fe
  
  # #normalize data
  # for (i in names(voice_fe)) {
  #   voice_fe_normalised[i] <- (voice_fe[i]*sapply(voice_fe[i],mean))/sapply(voice_fe[i],sd)
  # }
  
  voice_fe_normalised
  
  
  
}

#What is the voice i hear
what_is <- function(voice,model,type){
  if (type == "rf"){
    prediction <- predict(model, voice)
  } else{
    d <- process_live(voice)
    prediction <- predict(model, d)
  }
  
  if(type == "logistic" || type == "poly" ){
    #logistic and poly
    if(prediction > 0.5){
      print("Male")
    } else{
      print("Female")
    }
  }else if (type == "lda" || type == "qda"){
    
    #lda and qda 
    if(as.numeric(prediction$class) > 1){
      print("Male")
    } else{
      print("Female")
    }
  } else if (type == "svm"){
    #svm
    if(as.numeric(prediction) > 1){
      print("Male")
    } else{
      print("Female")
    }
  } else if (type == "rf"){
    #rf
    if(as.numeric(prediction) > 1){
      print("Male")
    } else{
      print("Female")
    }
  }else{
    print("Wrong paramters")
  }
  
}


#ROC function
roc_plot = function(actual,predicted){
  InformationValue::plotROC(actual, predicted)
  cat("AUC = ",InformationValue::AUROC(actual, predicted),"\n")
}


#Gain and Lift chart
gain_lift_plot = function(pred){
  par(mfrow=c(1,2))
  perf_lift <- performance(pred,"lift","rpp")
  perf_gain <- performance(pred,"tpr","rpp")
  plot(perf_lift,main="Lift Curve",col="red",lwd =2)
  plot(perf_gain,main="Gain Curve",col="orange",lwd=2)
}


#Kolmogorov-Smirnov plot
ks_chart = function(actual,predicted){
  ks <- ks_stat(actual, predicted)
  cat("KS-Statistic",ks,"\n")
  print(ks_stat(actual,predicted,returnKSTable = T))
  ks_plot(actual,predicted)
}



##############################################
####multi model builder

#accuracy for logistic
accuracy = function(data_set){
  
  names(data_set) <- c("X1","X2","X3","X4","X5","X6","X7","X8","lab")
  num <- as.integer(dim(data_set)[1]*0.8)
  train <- sample(dim(data_set)[1], num)
  
  
  #Validation data
  validation_data <- data_set[-train,]
  
  #Training data
  training_data <- data_set[train,]
  
  
  
  model1 <- glm(lab ~ ., data = training_data, family = "binomial")
  pred <- predict(model1, validation_data, type = "response")
  prediction <- rep(1, dim(validation_data)[1])
  prediction[pred<0.5] <- 0
  
  mse <- mean((prediction - validation_data$lab)**2)
  sum(prediction==validation_data$lab)/dim(validation_data)[1]
  
}

#cross validation function
#takes input of a dataframe with chosen predictors
#outputs the chosen subsets errors
#Regsubsets

cross_validate = function(method_name, train_data, split_percentage){
  
  #split size
  first <- 0
  second <- as.integer(split_percentage*dim(train_data)[1]/100)
  delta <- second - first
  
  #initialise variables
  pred_num <- dim(train_data)[2]-1
  final_error_list <- data.frame(matrix(nrow = as.integer(dim(train_data)[1]/delta), ncol = pred_num))
  error_list = rep(NA,pred_num)
  
  for (i in 1:as.integer(dim(train_data)[1]/delta)) {
    #kfold cross validation divide
    row1 <- train_data[0:first , ]
    row2 <- train_data[second:dim(train_data)[1],]
    test_set <- train_data[first:second, ]
    train_set <- data.frame(rbind(row1,row2))
    
    for (j in 1:pred_num) {
      fit <- regsubsets(lab ~. , data = train_set, nvmax = j, method = method_name)
      test_matrix <- model.matrix(lab ~., data = test_set)
      coefficient = coef(fit, j)
      
      pred = test_matrix[,names(coefficient)]%*%coefficient
      
      prediction <- rep(1, dim(test_set)[1])
      prediction[pred<0.5] <- 0
      error_list[j] = mean((test_set$lab - prediction )**2)
      
    }
    
    final_error_list[i,] <- error_list
    
    error_list = rep(NA,pred_num)
    first <- first + delta
    second <- second + delta
    
  }
  
  colMeans(final_error_list)
  
  
}


#pass to this the number of predictors and it will output a truth table
model_chooser = function(pred_num) {
  
  models <- data.frame(matrix(nrow = 3**pred_num, ncol = pred_num))
  n = pred_num
  
  #build truth table
  for( count in 1:pred_num) {
    
    build_1 <- rep("1", 3**(n-1))
    build_2 <- rep("x2", 3**(n-1))
    build_3 <- rep("0", 3**(n-1))
    
    build <- c(build_1,build_2,build_3)
    b <- rep(build,3**(pred_num)/length(build))
    models[,count] <- b
    n = n-1
  }
  
  models
}



model_builder = function(data) {
  
  pred_num = as.integer(dim(data)[2] - 1)
  truth_table <- model_chooser(pred_num)
  pred_index <- pred_num + 1
  predictor_names <- names(data[,-pred_index])
  
  #x x2 and x3 are the types a model can take
  x <- data[,-pred_index]
  x2 <- data[,-pred_index]**2
  x3 <- (data[,-pred_index])**3
  acc <- rep(NA, dim(truth_table)[1])
  
  #new model
  new_model <- data.frame(matrix(nrow = dim(data)[1], ncol = pred_num))
  
  #for each model combintaion that is saved
  for (n in 1:dim(truth_table)[1]) {
    mod <- truth_table[n,]
    #print(mod)
    counter <- 1
    for (ele in mod) {
      if (ele == "1") {
        new_model[,counter] <- x[,counter]
        counter <- counter+1
      } else if (ele == "x2") {
        new_model[,counter] <- x2[,counter]
        counter <- counter+1
      }else if (ele == "0") {
        new_model[,counter] <- 0
        counter <- counter+1
      }
      
    }
    
    new_model <- cbind(new_model, data$lab)
    
    #error for each model
    acc[n] <- accuracy(new_model)
    print(acc[n])
    new_model <- data.frame(matrix(nrow = dim(data)[1], ncol = pred_num))
    
  }
  
  final <- cbind(truth_table,acc)
  names(final) <- c("X1","X2","X3","X4","X5","X6","X7","X8","Accuracy")
  final
}

###############################################

#data split

voice_final <- process_data(voice)
voice_final
a <- 200
b <- dim(voice_final)[1]

#Put away this data in the safe.
safe_data <- tail(voice_final,a)


#Working data
work_data <- head(voice_final, b-a)

num <- as.integer(0.8*dim(work_data)[1])
train <- sample(dim(work_data)[1], num)

#Validation data
validation_data <- work_data[-train,]

#Training data
training_data <- work_data[train,]


#logistic

first_logistic_model <- glm(lab ~. , data = training_data, family = "binomial")
summary(first_logistic_model)
logistic_model <- final_logistic_model <- glm(lab ~ Mean.function.and.Std.dev + distribution + Skew.and.Kurt  +  Spectral.entropy.and.sfm + Mean.fun + Min.max.fun, data = training_data, family = "binomial")
summary(logistic_model)

pred <- predict(logistic_model, validation_data)
prediction <- rep(1, dim(validation_data)[1])
prediction[pred<0.5] <- 0
log_loss <- mean((validation_data$lab - prediction)**2)
log_acc <- sum(prediction==validation_data$lab)/dim(validation_data)[1]

confusion_matrix_plot(table(prediction,validation_data$lab),"Logistic Regression")
pred_gl =prediction(as.numeric(prediction),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, prediction)
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, prediction)



final_logistic_model <- glm(lab ~ Mean.function.and.Std.dev + distribution + Skew.and.Kurt  +  Spectral.entropy.and.sfm + Mean.fun + Min.max.fun, data = work_data, family = "binomial")
summary(final_logistic_model)



#logistic with bestsubset
model <- bestglm(training_data, family = binomial ,IC = "CV", method = "exhaustive")
model
logistic_model_bss <- glm(lab ~ Mean.function.and.Std.dev + distribution + Skew.and.Kurt + Spectral.entropy.and.sfm + Mean.fun + Min.max.fun , data = training_data, family = "binomial")
pred <- predict(logistic_model_bss, validation_data)
prediction <- rep(1, dim(validation_data)[1])
prediction[pred<0.5] <- 0
log_loss_bss <- mean((validation_data$lab - prediction)**2)
log_acc_bss <- sum(prediction==validation_data$lab)/dim(validation_data)[1]
final_logistic_model_bss <- glm(lab ~ Mean.function.and.Std.dev + distribution + Skew.and.Kurt + Spectral.entropy.and.sfm + Mean.fun + Min.max.fun , data = work_data, family = "binomial")

confusion_matrix_plot(table(prediction,validation_data$lab),"Logistic Regression with Best Subset")
pred_gl =prediction(as.numeric(prediction),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, prediction)
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, prediction)

summary(final_logistic_model_bss)


#logistic model with lasso

x <- model.matrix(lab~., data = training_data)
x_test <- model.matrix(lab~., data=validation_data)
y <- as.matrix(training_data[,c(9)])
cv.lassso <- cv.glmnet(x,training_data$lab,family = "binomial",alpha=1)
bestlambda <- cv.lassso$lambda.min
model_lasso <- glmnet(x, training_data$lab, family = "binomial", alpha = 1, lambda = bestlambda)
model_lasso
model_lasso$a0
model_lasso$beta

results <- predict(model_lasso, s=bestlambda, newx = x_test, type="response")
prediction <- rep(1, dim(x_test)[1])
prediction[results<0.5] <- 0
log_loss_lasso <- mean((validation_data$lab - prediction)**2)
log_acc_lasso <- sum(prediction==validation_data$lab)/dim(validation_data)[1]
plot(cv.lassso)
model_lasso$beta
final_logistic_model_lasso <- glm(lab ~. , data = work_data, family = "binomial")


confusion_matrix_plot(table(prediction,validation_data$lab),"Logistic Regression with Lasso")
pred_gl =prediction(as.numeric(prediction),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, prediction)
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, prediction)

#polynomial
poly <- model_builder(work_data)
max(poly$Accuracy)
poly[which.max(poly$Accuracy),]
poly_model <- glm(lab ~  distribution + Skew.and.Kurt  +  Spectral.entropy.and.sfm + Mode + Mean.fun + I(Min.max.fun^2) + I(Df.range^2) , data = training_data, family = "binomial") 
pred <- predict(poly_model, validation_data, type = "response")
prediction <- rep(1, dim(x_test)[1])
prediction[pred<0.5] <- 0
log_loss_poly <- mean((validation_data$lab - prediction)**2)
log_acc_poly <- sum(prediction==validation_data$lab)/dim(validation_data)[1]


confusion_matrix_plot(table(prediction,validation_data$lab),"Logistic Regression with Polynomial")
pred_gl =prediction(as.numeric(prediction),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, prediction)
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, prediction)


best_poly_model <- glm(lab ~  distribution + Skew.and.Kurt  +  Spectral.entropy.and.sfm + Mode + Mean.fun + I(Min.max.fun^2) + I(Df.range^2), data = work_data, family = "binomial") 




#lda
lda_model <- lda(lab~.,data= training_data)
plot(lda_model)
lda_predict <- predict(lda_model, validation_data)
lda_loss <- mean((validation_data$lab - (as.numeric(lda_predict$class)-1))**2)
lda_acc <- sum((lda_predict$class ==  validation_data$lab))/dim(validation_data)[1]
lda_final_model <- lda(lab~.,data= work_data)


confusion_matrix_plot(table(lda_predict$class,validation_data$lab),"LDA")
pred_gl =prediction(as.numeric(lda_predict$class),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, lda_predict$posterior[,2])
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, lda_predict$posterior[,2])


lda_loss
lda_acc
lda_model


#qda
qda_model <- qda(lab~.,data= training_data)
qda_predict <- predict(qda_model, validation_data)
qda_loss <-  mean((validation_data$lab - (as.numeric(qda_predict$class)-1))**2)
qda_acc <- sum((qda_predict$class ==  validation_data$lab))/dim(validation_data)[1]
qda_final_model <- qda(lab ~. , data = work_data)
#plot.roc(validation_data$lab,as.numeric(qda_predict$class), main = "QDA", ylim = c(0,1))
qda_acc
qda_loss
qda_model


confusion_matrix_plot(table(qda_predict$class,validation_data$lab),"QDA")
pred_gl =prediction(as.numeric(qda_predict$class),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, qda_predict$posterior[,2])
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, qda_predict$posterior[,2])



#svm

#svm linear

svm_model <- tune(svm,as.factor(lab)~. , data = training_data, kernel = "linear", ranges = list (cost = c(0.01,0.1,1,10,100)))
summary(svm_model)
svm_model <- svm(as.factor(lab) ~. , data = training_data, kernel = "linear", cost = 10)
prediction <- predict(svm_model, validation_data)
svm_loss <- mean((validation_data$lab - (as.numeric(prediction)-1))**2)
svm_acc <- mean((prediction == validation_data$lab))

svm_best_model <- svm(as.factor(lab) ~. , data = work_data, kernel = "linear", cost = 10)
svm_loss
svm_acc

confusion_matrix_plot(table(prediction,validation_data$lab),"SVM")
pred_gl =prediction(as.numeric(prediction),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, as.numeric(prediction))
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, as.numeric(prediction))

#svm radial
svm_model_radial <- tune(svm,as.factor(lab)~. , data = training_data, kernel = "radial", ranges = list (cost = c(0.01,0.1,1,10,100), gamma = c(0.01,0.1,1,10)))
summary(svm_model_radial)
svm_model_radial <- svm(as.factor(lab) ~. , data = training_data, kernel = "radial", cost = 10,gamma = 0.125 )
prediction <- predict(svm_model_radial, validation_data)
svm_loss_radial <- mean((validation_data$lab - (as.numeric(prediction)-1))**2)
svm_acc_radial <- mean((prediction == validation_data$lab))

svm_best_model_radial <- svm(as.factor(lab) ~. , data = work_data, kernel = "radial", cost = 10, gamma = 0.125)
svm_loss_radial
svm_acc_radial

confusion_matrix_plot(table(prediction,validation_data$lab),"SVM")
pred_gl =prediction(as.numeric(prediction),as.numeric(validation_data$lab))
roc_plot(validation_data$lab, as.numeric(prediction))
gain_lift_plot(pred_gl)
ks_chart(validation_data$lab, as.numeric(prediction))


#start analysis

#Female Kavya
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val1=c(0.2306929,0.03952188,0.2367605,0.2206177,0.2515580,0.03094029,2.376002,8.667609,0.8498063,0.2047239,0.2357996,0.2306929,0.2050574,0.04838710,0.2790698,0.9556108,0,8.390625,8.390625,0.03603810)
v1=data.frame(rbind(val1))
names(v1)=head

#male Jason
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val2=c(0.1899906,0.05179689,0.1768085,0.1608511,0.2348936,0.07404255,2.611782,11.631997,0.8946549,0.3591778,0.1746809,0.1899906,0.1342302,0.04701273,0.2758621,0.8475922,0,8.765625,8.765625,0.08793821)
v2=data.frame(rbind(val2))
names(v2)=head

#Male Sayed
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val3=c(0.1962980,0.05420681,0.2172776,0.1488532,0.2409861,0.09213290,2.243768,7.896143,0.8799352,0.2469445,0.1518542,0.1962980,0.1343376,0.04729064,0.2758621,1.1395089,0,8.296875,8.296875,0.10964912)
v3=data.frame(rbind(val3))
names(v3)=head

#Female Anusha
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val4=c(0.2134637,0.04141575,0.2233049,0.2085337,0.2330799,0.02454616,3.647884,18.30468,0.8467511,0.2744759,0.228301,0.2134637,0.1686301,0.047105,0.2758621,0.6860795,0,7.992188,7.992188,0.07900638)
v4=data.frame(rbind(val4))
names(v4)=head

#Female Alekhya
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val5=c(0.2091013,0.06415903,0.2299235,0.1720529,0.2556437,0.08359081,2.185829,7.697031,0.8942238,0.3577541,0.222714,0.2091013,0.1843127,0.048,0.2790698,0.7888064,0,7.992188,7.992188,0.02720756)
v5=data.frame(rbind(val5))
names(v5)=head


#Male Mahesh
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val6=c(0.169223,0.0564518,0.141825,0.1329609,0.2158659,0.08290503,3.573528,17.67989,0.8795342,0.34232,0.1350466,0.169223,0.1077725,0.04696673,0.2790698,1.742996,0.0234375,15.84375,15.82031,0.1214298)
v6=data.frame(rbind(val6))
names(v6)=head


#Female Sujatha
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val7=c(0.1900508,0.03485598,0.1915209,0.1807092,0.2071868,0.02647754,3.412758,17.81338,0.8410903,0.2438406,0.1868873,0.1900508,0.1512332,0.04696673,0.2711864,0.742086,0,11.625,11.625,0.03029717)
v7=data.frame(rbind(val7))
names(v7)=head



#Female Sneha
head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
val8=c(0.2109933,0.06356805,0.2325959,0.1810835,0.2575621,0.07647856,2.634728,11.62647,0.9023707,0.4580564,0.2743115,0.2109933,0.1777395,0.04692082,0.2774566,1.627651,0,7.171875,7.171875,0.1100629)
v8=data.frame(rbind(val8))
names(v8)=head


list_of_voices <- rbind(v1,v2,v3,v4,v5,v6,v7,v8)
list_of_voices <- cbind(list_of_voices, c("Female", "Male", "Male", "Female","Female","Male","Female","Female"))
list_of_voices[,21] <- as.character(list_of_voices[,21])

answers <- data.frame(matrix(NA,nrow = dim(list_of_voices)[1],ncol = 8))
for (i in 1:dim(list_of_voices)[1]){
  processed_voice <- list_of_voices[i,]
  
  logis <- what_is(processed_voice,final_logistic_model,"logistic")
  bss <- what_is(processed_voice,final_logistic_model_bss,"logistic")
  lasso <- what_is(processed_voice,final_logistic_model_lasso,"logistic")
  poly <- what_is(processed_voice,best_poly_model,"poly")
  lda <- what_is(processed_voice,lda_final_model,"lda")
  qda <- what_is(processed_voice,qda_final_model,"qda")
  svm <- what_is(processed_voice,svm_best_model,"svm")
  
  answers[i,] <- c("Logistic" = logis, "Best subset" = bss, "Lasso" = lasso, "Polynomial" = poly, "LDA" = lda, "QDA" = qda, "SVM" = svm, "Actual" = list_of_voices[i,21])
  
}
names(answers) <- c("Logistic", "Best subset", "Lasso", "Polynomial", "LDA", "QDA", "SVM", "Actual")
plot.new()
grid.table(answers)




##############################################################
#decrease data to see model accuracy
loss_vs_train_data_normal = function(data) {
  
  
  total <- dim(data)[1]
  #total <- 52
  error_list <- rep(NA,dim(data)[1])
  j = 1
  data <- data[sample(nrow(data),nrow(data)),]
  #for (i in seq(from = as.numeric(1/100*total) , to = as.numeric(99/100*total), by = as.numeric(1/100*total))) {
  for (i in seq(from = 1, to = dim(data)[1], by = 20)) {
    
    #val <- as.numeric(total-as.integer(i/100*total))
    
    
    train_data <- data[c(1:i),]
    
    test_data <- data[c(i:total),]
    
    model <- glm(lab ~ ., data = train_data, family = "binomial")
    pred <- predict(model , test_data, type = "response")
    prediction <- rep(1, dim(test_data)[1])
    prediction[pred<0.5] <- 0
    
    error_list[i] <- mean(prediction==test_data$lab)
    j = j+1
    print(error_list[i])
    
  }
  error_list
}



err <- loss_vs_train_data_normal(work_data)
length(err)
split_vector <- seq(from = 1,to= length(err))
plot(split_vector,err, type = "b", ylim = c(0.8,1))
#############################################################################################################

Methods <- c("Logistic", "Best Subset","Lasso","Polynomial","LDA","QDA","SVM")

#validation loss and acc

Validation_loss <- c(log_loss,log_loss_bss,log_loss_lasso,log_loss_poly,lda_loss, qda_loss,svm_loss)
Validation_accuracy <- 100*c(log_acc,log_acc_bss,log_acc_lasso,log_acc_poly,lda_acc,qda_acc,svm_acc)

Validation_loss <- round(Validation_loss,3)
Validation_accuracy <- round(Validation_accuracy,2)

Validation_loss <- rbind(Methods,Validation_loss)
Validation_accuracy <- rbind(Methods,Validation_accuracy)

plot.new()
grid.table(Validation_loss)
plot.new()
grid.table(Validation_accuracy)



#Safe accuracy

safe_log <- predict(final_logistic_model, safe_data)
prediction <- rep(1, dim(safe_data)[1])
prediction[safe_log<0.5] <- 0
safe_log_loss <- mean((safe_data$lab - prediction)**2)
safe_log_acc<- sum(prediction==safe_data$lab)/dim(safe_data)[1]


safe_log_bss <- predict(final_logistic_model_bss, safe_data)
prediction <- rep(1, dim(safe_data)[1])
prediction[safe_log_bss<0.5] <- 0
safe_log_loss_bss <- mean((safe_data$lab - prediction)**2)
safe_log_acc_bss<- sum(prediction==safe_data$lab)/dim(safe_data)[1]


safe_log_lasso <- predict(final_logistic_model_lasso, safe_data)
prediction <- rep(1, dim(safe_data)[1])
prediction[safe_log_lasso<0.5] <- 0
safe_log_loss_lasso <- mean((safe_data$lab - prediction)**2)
safe_log_acc_lasso<- sum(prediction==safe_data$lab)/dim(safe_data)[1]


safe_log_poly <- predict(best_poly_model, safe_data)
prediction <- rep(1, dim(safe_data)[1])
prediction[safe_log_poly<0.5] <- 0
safe_log_loss_poly <- mean((safe_data$lab - prediction)**2)
safe_log_acc_poly<- sum(prediction==safe_data$lab)/dim(safe_data)[1]


safe_lda_predict <- predict(lda_final_model, safe_data)
safe_lda_loss <- mean((safe_data$lab - (as.numeric(safe_lda_predict$class)-1))**2)
safe_lda_acc <- sum((safe_lda_predict$class ==  safe_data$lab))/dim(safe_data)[1]


safe_qda_predict <- predict(qda_final_model, safe_data)
safe_qda_loss <- mean((safe_data$lab - (as.numeric(safe_qda_predict$class)-1))**2)
safe_qda_acc <- sum((safe_qda_predict$class ==  safe_data$lab))/dim(safe_data)[1]



svm_best_model
prediction <- predict(svm_best_model, safe_data)
safe_svm_loss <- mean((safe_data$lab - (as.numeric(prediction)-1))**2)
safe_svm_acc <- mean((prediction == safe_data$lab))


#Safe log and loss
safe_loss <- c(safe_log_loss,safe_log_loss_bss,safe_log_loss_lasso,safe_log_loss_poly,safe_lda_loss, safe_qda_loss,safe_svm_loss)
safe_accuracy <- 100*c(safe_log_acc,safe_log_acc_bss,safe_log_acc_lasso,safe_log_acc_poly,safe_lda_acc,safe_qda_acc,safe_svm_acc)

safe_loss <- round(safe_loss,3)
safe_accuracy <- round(safe_accuracy,2)

safe_loss <- rbind(Methods,safe_loss)
safe_accuracy <- rbind(Methods,safe_accuracy)

plot.new()
grid.table(safe_loss)
plot.new()
grid.table(safe_accuracy)



######=================================== FOREST Begins ==========================================================================


set.seed(24)
#load data
voice <- data.frame(filePath)

voice_input <- voice[,-21]
voice_label <- data.frame(voice[,21])
names(voice_label)="label"

voice_fe_normalised=voice_input
# for (i in names(voice_input)) {
#   voice_fe_normalised[i] <- (voice_input[i]*sapply(voice_input[i],mean))/sapply(voice_input[i],sd)
# }

voice=data.frame(cbind(voice_fe_normalised,voice_label))

train=sample(1:nrow(voice),2400)
voice_train=voice[train,]


voice_validate_split=voice[-train,]
validate=sample(1:nrow(voice_validate_split),384)
voice_validate=voice_validate_split[validate,]
voice_test=voice_validate_split[-validate,]


zeroOne=function(inp)
{
  label=rep(999,length(inp))
  for (i in 1:length(inp))
  {
    if(inp[i] == "male"){
      label[i]=1
    }
    else
    {
      label[i]=0
    }
  }
  label
}

######tree=====================================================================================

tree_output=tree(label~.,voice_train)
summary(tree_output)
tree_prediction=predict(tree_output,voice_validate,type="class")


##evaluating tree
prediction_tree=rep("female",nrow(voice_validate))
prediction_tree[tree_prediction == "male"]="male"
mean(voice_validate$label==prediction_tree)

pred=prediction(as.numeric(tree_prediction),as.numeric(voice_validate$label))
predict_conversion=zeroOne(prediction_tree)
dataLabel_conversion_tree=zeroOne(voice_validate$label)

confusion_matrix_plot(table(predict_conversion, dataLabel_conversion_tree),"Decision tree")
roc_plot(dataLabel_conversion_tree,predict_conversion)
gain_lift_plot(pred)
ks_chart(dataLabel_conversion_tree, predict_conversion)


plot(tree_output,main="Tree")
text(tree_output,pretty=0)

####=============================



#Bagging  ============================================================

bagging_fit=randomForest(label~.,data=voice_train,mtry=20,importance=TRUE)
summary(bagging_fit)
bag_predict = predict (bagging_fit ,voice_validate)
summary(bag_predict)

prediction_bag=rep("female",nrow(voice_validate))
prediction_bag[bag_predict == "male"]="male"
mean(voice_validate$label==prediction_bag)

pred=prediction(as.numeric(bag_predict),as.numeric(voice_validate$label))

predict_conversion=zeroOne(prediction_bag)
dataLabel_conversion_tree=zeroOne(voice_validate$label)

confusion_matrix_plot(table(predict_conversion, dataLabel_conversion_tree),"Bagging")
roc_plot(dataLabel_conversion_tree,predict_conversion)
gain_lift_plot(pred)
ks_chart(dataLabel_conversion_tree, predict_conversion)


#evaluating bagging model
importance(bagging_fit)
varImpPlot(bagging_fit)


#####==================================================================================


#Random Forest ============================================================

random_forest=function(voice_train,voice_validate){
  #acc_martrix=rep(NA,20)
  
  cols=ncol(voice_train)-1
  num_tree=c(25,50,75,100,200)
  acc_martrix=matrix(c(1:cols),c(1:5),ncol=20,nrow=5)
  
  
  for (j in 1:nrow(acc_martrix))
  {
    for(i in 1:ncol(acc_martrix))
    {
      forest_fit=randomForest(label~.,data=voice_train,mtry=i,importance=TRUE,ntree=num_tree[j])
      forest_predict = predict (forest_fit ,voice_validate)
      
      #evaluating Random Forest
      prediction_forest=rep("female",nrow(voice_validate))
      prediction_forest[forest_predict == "male"]="male"
      acc=mean(voice_validate$label==prediction_forest)
      acc_martrix[j,i]=acc
    }
  }
  acc_martrix
  
}
acc=random_forest(voice_train,voice_validate)
acc
num_tree=c(25,50,75,100,200)
max_acc=which(acc == max(acc), arr.ind = TRUE)
max_acc


forest_fit_final=randomForest(label~.,data=voice_train,mtry=max_acc[1,2],ntree=num_tree[max_acc[1,1]],importance=TRUE)
forest_fit_final
forest_predict_final = predict (forest_fit_final ,voice_validate)
summary(forest_predict_final)

#evaluating Random Forest
prediction_forest=rep("female",nrow(voice_validate))
prediction_forest[forest_predict_final == "male"]="male"
mean(voice_validate$label==prediction_forest)



predict_conversion=zeroOne(forest_predict_final)
dataLabel_conversion_tree=zeroOne(voice_validate$label)

pred=prediction(as.numeric(forest_predict_final),as.numeric(voice_validate$label))

confusion_matrix_plot(table(predict_conversion, dataLabel_conversion_tree),"Random Forest")
roc_plot(dataLabel_conversion_tree,predict_conversion)
gain_lift_plot(pred)
ks_chart(dataLabel_conversion_tree, predict_conversion)


importance(forest_fit_final)
varImpPlot(forest_fit_final)

###================================================

### safe data accuracy evaluation:=========================


##Decision Tree

tree_prediction_safe=predict(tree_output,voice_test,type="class")
##evaluating tree
prediction_tree=rep("female",nrow(voice_test))
prediction_tree[tree_prediction_safe == "male"]="male"
mean(voice_test$label==prediction_tree)



### Bagging
bag_predict = predict (bagging_fit ,voice_test)
summary(bag_predict)

prediction_bag=rep("female",nrow(voice_test))
prediction_bag[bag_predict == "male"]="male"
mean(voice_test$label==prediction_bag)


### random Forest
forest_predict_final = predict (forest_fit_final ,voice_test)
summary(forest_predict_final)

#evaluating Random Forest
prediction_forest=rep("female",nrow(voice_test))
prediction_forest[forest_predict_final == "male"]="male"
mean(voice_test$label==prediction_forest)

###======= FOREST ENDS===================================================================================================================





###### RealTime prediction ================================================================================================
set.seed(1)

voice <- data.frame(filePath)
voice_input <- voice[,-21]
voice_label <- data.frame(voice[,21])
names(voice_label) <- c("Gender")
names(voice)

processFolder <- function(folderName) {
  # Start with empty data.frame.
  data <- data.frame()
  
  # Get list of files in the folder.
  list <- list.files(folderName, '\\.wav')
  
  # Add file list to data.frame for processing.
  for (fileName in list) {
    row <- data.frame(fileName, 0, 0, 20)
    data <- rbind(data, row)
  }
  
  # Set column names.
  names(data) <- c('sound.files', 'selec', 'start', 'end')
  
  # Move into folder for processing.
  setwd(folderName)
  
  # Process files.
  acoustics <- specan3(data, parallel=1)
  
  # Move back into parent folder.
  setwd('..')
  
  acoustics
}


specan3 <- function(X, bp = c(0,22), wl = 2048, threshold = 5, parallel = 1){
  # To use parallel processing: library(devtools), install_github('nathanvan/parallelsugar')
  if(class(X) == "data.frame") {if(all(c("sound.files", "selec", 
                                         "start", "end") %in% colnames(X))) 
  {
    start <- as.numeric(unlist(X$start))
    end <- as.numeric(unlist(X$end))
    sound.files <- as.character(unlist(X$sound.files))
    selec <- as.character(unlist(X$selec))
  } else stop(paste(paste(c("sound.files", "selec", "start", "end")[!(c("sound.files", "selec", 
                                                                        "start", "end") %in% colnames(X))], collapse=", "), "column(s) not found in data frame"))
  } else  stop("X is not a data frame")
  
  #if there are NAs in start or end stop
  if(any(is.na(c(end, start)))) stop("NAs found in start and/or end")  
  
  #if end or start are not numeric stop
  if(all(class(end) != "numeric" & class(start) != "numeric")) stop("'end' and 'selec' must be numeric")
  
  #if any start higher than end stop
  if(any(end - start<0)) stop(paste("The start is higher than the end in", length(which(end - start<0)), "case(s)"))  
  
  #if any selections longer than 20 secs stop
  if(any(end - start>20)) stop(paste(length(which(end - start>20)), "selection(s) longer than 20 sec"))  
  options( show.error.messages = TRUE)
  
  #if bp is not vector or length!=2 stop
  if(!is.vector(bp)) stop("'bp' must be a numeric vector of length 2") else{
    if(!length(bp) == 2) stop("'bp' must be a numeric vector of length 2")}
  
  #return warning if not all sound files were found
  fs <- list.files(path = getwd(), pattern = ".wav$", ignore.case = TRUE)
  if(length(unique(sound.files[(sound.files %in% fs)])) != length(unique(sound.files))) 
    cat(paste(length(unique(sound.files))-length(unique(sound.files[(sound.files %in% fs)])), 
              ".wav file(s) not found"))
  
  #count number of sound files in working directory and if 0 stop
  d <- which(sound.files %in% fs) 
  if(length(d) == 0){
    stop("The .wav files are not in the working directory")
  }  else {
    start <- start[d]
    end <- end[d]
    selec <- selec[d]
    sound.files <- sound.files[d]
  }
  
  # If parallel is not numeric
  if(!is.numeric(parallel)) stop("'parallel' must be a numeric vector of length 1") 
  if(any(!(parallel %% 1 == 0),parallel < 1)) stop("'parallel' should be a positive integer")
  
  # If parallel was called
  if(parallel > 1)
  { options(warn = -1)
    if(all(Sys.info()[1] == "Windows",requireNamespace("parallelsugar", quietly = TRUE) == TRUE)) 
      lapp <- function(X, FUN) parallelsugar::mclapply(X, FUN, mc.cores = parallel) else
        if(Sys.info()[1] == "Windows"){ 
          cat("Windows users need to install the 'parallelsugar' package for parallel computing (you are not doing it now!)")
          lapp <- pbapply::pblapply} else lapp <- function(X, FUN) parallel::mclapply(X, FUN, mc.cores = parallel)} else lapp <- pbapply::pblapply
  
  options(warn = 0)
  
  if(parallel == 1) cat("Measuring acoustic parameters:")
  x <- as.data.frame(lapp(1:length(start), function(i) { 
    r <- tuneR::readWave(file.path(getwd(), sound.files[i]), from = start[i], to = end[i], units = "seconds") 
    
    b<- bp #in case bp its higher than can be due to sampling rate
    if(b[2] > ceiling(r@samp.rate/2000) - 1) b[2] <- ceiling(r@samp.rate/2000) - 1 
    
    
    #frequency spectrum analysis
    songspec <- seewave::spec(r, f = r@samp.rate, plot = FALSE)
    analysis <- seewave::specprop(songspec, f = r@samp.rate, flim = c(0, 280/1000), plot = FALSE)
    
    #save parameters
    meanfreq <- analysis$mean/1000
    sd <- analysis$sd/1000
    median <- analysis$median/1000
    Q25 <- analysis$Q25/1000
    Q75 <- analysis$Q75/1000
    IQR <- analysis$IQR/1000
    skew <- analysis$skewness
    kurt <- analysis$kurtosis
    sp.ent <- analysis$sh
    sfm <- analysis$sfm
    mode <- analysis$mode/1000
    centroid <- analysis$cent/1000
    
    
    #Frequency with amplitude peaks
    peakf <- 0#seewave::fpeaks(songspec, f = r@samp.rate, wl = wl, nmax = 3, plot = FALSE)[1, 1]
    
    #Fundamental frequency parameters
    ff <- seewave::fund(r, f = r@samp.rate, ovlp = 50, threshold = threshold, 
                        fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = wl)[, 2]
    meanfun<-mean(ff, na.rm = T)
    minfun<-min(ff, na.rm = T)
    maxfun<-max(ff, na.rm = T)
    
    #Dominant frecuency parameters
    y <- seewave::dfreq(r, f = r@samp.rate, wl = wl, ylim=c(0, 280/1000), ovlp = 0, plot = F, threshold = threshold, bandpass = b * 1000, fftw = TRUE)[, 2]
    meandom <- mean(y, na.rm = TRUE)
    mindom <- min(y, na.rm = TRUE)
    maxdom <- max(y, na.rm = TRUE)
    dfrange <- (maxdom - mindom)
    duration <- (end[i] - start[i])
    
    #modulation index calculation
    changes <- vector()
    for(j in which(!is.na(y))){
      change <- abs(y[j] - y[j + 1])
      changes <- append(changes, change)
    }
    if(mindom==maxdom) modindx<-0 else modindx <- mean(changes, na.rm = T)/dfrange
    
    #save results
    return(c(duration, meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, 
             centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx))
  }))
  
  #change result names
  rownames(x) <- c("duration", "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", 
                   "sfm","mode", "centroid", "peakf", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx")
  x <- data.frame(sound.files, selec, as.data.frame(t(x)))
  colnames(x)[1:2] <- c("sound.files", "selec")
  rownames(x) <- c(1:nrow(x))
  
  return(x)
}

#Path for MP3 File
mp3Path=mp3_input_path

#Mp3 to wav conversion
list <- list.files(mp3Path, '\\.mp3')
i=0

for (item in list){
  i=i+1
  tmp_Path=paste(mp3Path,item,sep="")
  r = readMP3(tmp_Path)
  setwd(mp3Path)
  tmp_Name=paste(i, ".wav", sep="")
  writeWave(r,tmp_Name,extensible=TRUE)
}

wavPath=mp3_input_path
voice_predict=processFolder(wavPath)

### Wait for completion
summary(voice_predict)

head=c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent", "sfm", "mode","centroid", "meanfun",    "minfun" ,   "maxfun",   "meandom", "mindom",   "maxdom",  "dfrange",   "modindx")
cutTempFields=voice_predict[4:24]
fields=cutTempFields[-13]
voiceFrame=data.frame(rbind(fields))
names(voiceFrame)=head
voiceFrame



### RealTime Prediction ends =================================================================================


list_of_voices <- cbind(voiceFrame, c("Female","Male"))
list_of_voices[,21] <- as.character(list_of_voices[,21])

answers2 <- data.frame(matrix(NA,nrow = dim(list_of_voices)[1],ncol = 9))
for (i in 1:dim(list_of_voices)[1]){
  processed_voice <- list_of_voices[i,]
  
  logis <- what_is(processed_voice,final_logistic_model,"logistic")
  bss <- what_is(processed_voice,final_logistic_model_bss,"logistic")
  lasso <- what_is(processed_voice,final_logistic_model_lasso,"logistic")
  poly <- what_is(processed_voice,best_poly_model,"poly")
  lda <- what_is(processed_voice,lda_final_model,"lda")
  qda <- what_is(processed_voice,qda_final_model,"qda")
  svm <- what_is(processed_voice,svm_best_model,"svm")
  rf <- what_is(processed_voice, forest_fit_final, "rf")
  answers2[i,] <- c("Logistic" = logis, "Best subset" = bss, "Lasso" = lasso, "Polynomial" = poly, "LDA" = lda, "QDA" = qda, "SVM" = svm, "RF" = rf, "Actual" = list_of_voices[i,21])
  
}
names(answers2) <- c("Logistic", "Best subset", "Lasso", "Polynomial", "LDA", "QDA", "SVM", "RF", "Actual")
plot.new()
grid.table(answers2)

######=======================================================================================







