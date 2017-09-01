# euclidean loss visuals

mydata <- read.table("C:\\Users\\yue\\Documents\\overview\\1-process-data\\final\\valid\\losses.txt", header=TRUE, 
  	sep="\t", as.is=TRUE)
dim(mydata)
colnames(mydata) <- c('image_name', 'num_true_coords', 'num_preds', 'norm_euclid_distance', 'num_preds_all', 'norm_euclid_distance_all');

colnames(mydata) 

plot(mydata[,2], mydata[,3], xlab='True # Preds', ylab='Predicted # Preds', main='True Number of Coords vs Predicted Number of Preds');
grid();

library(BlandAltmanLeh)
bland.altman.plot(mydata[,3], mydata[,2], main="Bland Altman of Difference in Predicted vs True Number of Cones", xlab="Means", ylab="Differences")
# library(ggplot2)
#pl <- bland.altman.plot(mydata[,2], mydata[,3], graph.sys = "ggplot2")
#print(pl)

boxplot(mydata[, 4])
title('Average Min Distance in Pixels')