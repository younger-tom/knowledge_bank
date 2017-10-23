library(xgboost)
library(e1071)
library(ROCR)
library(pROC)
library(Matrix)
source('/home/huangtao07/code/util.R')

data<-read.table("case170411_170426_all.txt",sep='\t',na.strings="NULL")
name<-read.table("var_all_transfer_2_0",sep='\t',na.strings="NULL")
colnames(data)<-name[,1]

data[,"label1"]<-0
data[which(data[,"status"]==10),"label1"]<-1
data[which(data[,"status"]==14),"label1"]<-1

train_data<-data[,-match("sessionid",colnames(data))]
train_data<-apply(train_data,2,as.numeric)

###simple 01 depth=5
dtrain<-xgb.DMatrix(as.matrix(train_data[which(train_data[,"status"]!=14),!(colnames(train_data) %in% c('sessionid','simple_transfer','extend_transfer','extend_transfer_2_0','status','label1'))]),label=as.matrix(train_data[which(train_data[,"status"]!=14),"label1"]),missing=NA)

watchlist<-list(eval=dtrain)

#without grey
model_01<-xgb.train(data=dtrain,max.depth=5,nround=1,objective = "binary:logistic",missing=NA,eval_metric="auc",eta=0.1,verbose=1,watchlist=watchlist)

xgb.fmap = genFMap(data.frame(as.matrix(train_data[which(train_data[,"status"]!=14),!(colnames(train_data) %in% c('sessionid','simple_transfer','extend_transfer','extend_transfer_2_0','status','label1'))])[1:2,]), "model_01.fmap")

xgb.dump(model_01, "model_01.txt",with.stats=T,fmap='model_01.fmap')

predict_leaf_01<-predict(model_01,newdata=dtrain,missing=NA,predleaf=TRUE)
data1<-(as.matrix(train_data[which(train_data[,"status"]!=14),]))     
data1<-cbind(data1,predict_leaf_01)
colnames(data1)<-c(colnames(train_data),"leaf")

result_01<-table(data1[,"leaf"],data1[,"label1"])
result_01<-cbind(as.matrix(result_01),as.matrix(result_01[,"1"]/(result_01[,"0"]+result_01[,"1"])))
write.table(result_01,"result_01",row.names=T,col.names=T,quote=F,sep='\t')
####
#write.table(data1,"data1.txt",row.names=F,col.names=F,quote=F,sep=',')
#write.table(colnames(data1),"varnames_data1.txt",row.names=F,col.names=F,quote=F)


###simple 02 depth=4
#without grey
model_02<-xgb.train(data=dtrain,max.depth=4,nround=1,objective = "binary:logistic",missing=NA,eval_metric="auc",eta=0.1,verbose=1,watchlist=watchlist)

xgb.fmap = genFMap(data.frame(as.matrix(train_data[which(train_data[,"status"]!=14),!(colnames(train_data) %in% c('sessionid','simple_transfer','extend_transfer','extend_transfer_2_0','status','label1'))])[1:2,]), "model_02.fmap")

xgb.dump(model_02, "model_02.txt",with.stats=T,fmap='model_02.fmap')

predict_leaf_02<-predict(model_02,newdata=dtrain,missing=NA,predleaf=TRUE)
data1<-(as.matrix(train_data[which(train_data[,"status"]!=14),]))     
data1<-cbind(data1,predict_leaf_02)
colnames(data1)<-c(colnames(train_data),"leaf")

result_02<-table(data1[,"leaf"],data1[,"label1"])
result_02<-cbind(as.matrix(result_02),as.matrix(result_02[,"1"]/(result_02[,"0"]+result_02[,"1"])))
write.table(result_02,"result_02",row.names=T,col.names=T,quote=F,sep='\t')
