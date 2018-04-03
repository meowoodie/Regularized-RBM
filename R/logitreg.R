# R code for exploring the correlations between keywords and crime series
# by using logistic regression with L1 regularization penalty. Specifically
# we applied logistic regression on each crime serie against other random
# crime cases, and explore the contribution of each keywords to discriminate
# different crime series.
# 
# By Shixiang Zhu
# Contact: shixiang.zhu@gatech.edu

library(glmnet)

# load raw data from local file
data.path  = "/Users/woodie/Desktop/workspace/Event2vec/resource/embeddings/2k.bigram.doc.tfidf.vecs.txt"
vocab.path = "/Users/woodie/Desktop/workspace/Event2vec/resource/vocab_list.txt"
rawdata = read.csv(data.path, header = FALSE, sep = ",")
vocab   = read.csv(vocab.path, header = FALSE, sep = " ")
n_sample = 200

set.seed(9)
# labeled rows indices
burglary            = 1:22
pedrobbery          = 23:26
dijawan_adams       = 27:34
jaydarious_morrison = 35:41
julian_tucker       = 42:48
thaddeus_todd       = 49:56
# select a subset of raw data, in particular, first 56 cases have been labeled
positive.x = rawdata[thaddeus_todd, ]
negative.x = rawdata[57:2056, ]
negative.x = negative.x[sample(nrow(negative.x), n_sample), ]
x = rbind(positive.x, negative.x)
x = x[, colSums(x != 0) > -1]
x = as.matrix(x)
# rename columns' name of matrix x
colnames(x) = apply(as.matrix(1:ncol(x)), 1, 
                    function(e){as.character(vocab$V2[match(e-1, vocab$V1)])})
# normalization for columes whose number of nonzero values is greater than 0
for (col in 1:ncol(x)){
  if (sum(x[,col] != 0) != 0){
    x[,col] = x[,col] / sum(x[,col])
  }
}
# labeling information
y = c(rep(1, nrow(positive.x)), rep(0, nrow(negative.x)))

# fit in lasso
cv.fit = cv.glmnet(x, y, family="binomial", alpha=1)
# get nonzero coefficients and their corresponding values
selection         = coef(cv.fit, s="lambda.min")
nonzero.selection = which(selection > 0) - 2
coef.selection    = selection[nonzero.selection + 2]
selected.words    = as.character(vocab$V2[match(nonzero.selection, vocab$V1)])
print(selected.words)
print(coef.selection)

# plot result
mod           = glmnet(x, y)
glmcoef       = coef(mod, cv.fit$lambda.min)
coef.increase = dimnames(glmcoef[glmcoef[,1]>0,0])[[1]]
coef.decrease = dimnames(glmcoef[glmcoef[,1]<0,0])[[1]]
# get ordered list of variables as they appear at smallest lambda
allnames = names(coef(mod)[,ncol(coef(mod))][order(coef(mod)[,ncol(coef(mod))],decreasing=TRUE)])
# remove intercept
allnames = setdiff(allnames,allnames[grep("Intercept",allnames)])
# assign colors
cols = rep("gray", length(allnames))
cols[allnames %in% coef.increase] = "red"      # higher mpg is good
cols[allnames %in% coef.decrease] = "blue"     # lower mpg is not

library(plotmo)
plot_glmnet(cv.fit$glmnet.fit, label=TRUE, s=cv.fit$lambda.min, col=cols)

