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
voice_input <- voice[,-21]
voice_label <- data.frame(voice[,21])


voice_fe_normalised=voice_input

#normalize data
for (i in names(voice_input)) {
  voice_fe_normalised[i] <- (voice_input[i]*sapply(voice_input[i],mean))/sapply(voice_input[i],sd)
}
voice_fe_normalised=voice_fe_normalised[,-7:-8]



lab <- rep(9,as.integer(dim(voice_label)[1]))

# 1 - male
# 0 - female
for (i in c(1:dim(voice_label)[1])){
  if(voice_label[i,1] == "male"){
    lab[i] = 1
  } else{
    lab[i] = 2
  }
}
vb=data.frame(cbind(voice_input,lab))


m=kmeans(voice_fe_normalised,2)
plot(voice_fe_normalised,col=m$cluster)

table(vb$lab,m$cluster)
