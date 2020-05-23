library('jackstraw')
library(ClusterR)
require(RcppCNPy)

set.seed(1234)
#dat = t(scale(t(Jurkat293T), center=TRUE, scale=FALSE))

folder = "../results/"
model_name = "model_Human_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_100_loss_combine00052_bysample_epoch10_best_separatelayer_rc_shiftleft_shiftright/"
filename = paste(folder, "motifs/keras_version/frozen/", model_name, "well_predicted/similarity_to_mouse_trim02outlier.npy", sep = "") # rows are for Human and columns are for Mouse
dat <- npyLoad(filename)
df <- as.data.frame(dat)
max_cluster = nrow(dat) - 1

dat[is.nan(dat)] <- 0
dat[is.infinite(dat)] <- 0

kmeans.dat <- KMeans_rcpp(dat, clusters = 10, num_init = 1,
                          max_iters = 100, initializer = 'kmeans++')

jackstraw.out <- jackstraw_kmeanspp(dat, kmeans.dat) #, s=30, B=100
# jackstraw.out <- jackstraw_pca(dat, r=1)

pvalue = unlist(jackstraw.out["p.F"])
num_sig_pvalue = sum(pvalue < 0.05)

PA_result <- permutationPA(dat, B = 30, threshold = 0.05, verbose = TRUE)

# Find the # clusters
library(factoextra)
library(NbClust)

# Standardize the data
#df_scaled <- scale(USArrests)
df_scaled <- scale(dat)

# Elbow method
fviz_nbclust(df, kmeans, method = "wss", k.max = max_cluster)+
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method", cex.axis=0.1)+
  theme(axis.text.x = element_text(size = 3))

# Silhouette method
fviz_nbclust(df, kmeans, method = "silhouette", k.max = max_cluster)+
  labs(subtitle = "Silhouette method")+
  theme(axis.text.x = element_text(size = 3))

# Gap statistic
fviz_nbclust(df, kmeans, nstart = 25,  method = "gap_stat",  k.max = max_cluster, nboot = 50)+
  labs(subtitle = "Gap statistic method") + theme(axis.text.x = element_text(size = 3))

# Majority rule
res.nbclust <- NbClust(df, distance = "euclidean", #df_scaled
                       min.nc = 2, max.nc = max_cluster, 
                       method = "kmeans", index ="all") #complete

factoextra::fviz_nbclust(res.nbclust) + theme_minimal() + ggtitle("NbClust's optimal number of clusters")