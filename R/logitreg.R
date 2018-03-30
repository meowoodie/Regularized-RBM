library(glmnet)
data.path = "/Users/woodie/Desktop/workspace/Event2vec/resource/embeddings/2k.bigram.doc.tfidf.vecs.txt"

rawdata = read.csv(data.path, header = FALSE, sep = ",")
x = rawdata[1:1022, ]
x = x[, colSums(x != 0) > -1]
x = as.matrix(x)
x = t(t(x+(1e-10))/colSums(x))
y = c(rep(1, 22), rep(0, 1000))

# fit = glmnet(x, y, family="binomial", alpha=1)
# summary(fit)
# 
# # Plot variable coefficients vs. shrinkage parameter lambda.
plot(fit, xvar="lambda")

cv.fit = cv.glmnet(x, y, family="binomial", alpha=1)
plot(cv.fit)

selection         = coef(cv.fit, s="lambda.min")
nonzero.selection = which(selection>0) - 2
coef.selection    = selection[nonzero.selection + 2]

# burglary, dusted, driveway, stolen_items, rear_door, jewelry, garage_door, master_bedroom, drawer, afis, attempt, quality, later_turned, pried, drawers, 