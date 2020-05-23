source("utils_R.R")
library(seqLogo) 

###################################################
### makePWM
###################################################
# mFile <- system.file("Exfiles/pwm1", package="seqLogo")
# m <- read.table(mFile)
# m
# p <- makePWM(m)

folder <- "..results/pytorch_version/Sasha_model2/" 
filename <- paste(folder, "filter_motifs_pwm.meme", sep = "")
output_folder <- paste(folder, "pwm_logos/", sep = "")

if (!dir.exists(file.path(output_folder))){dir.create(output_folder)}

memeMatrix <- read_meme(filename)

library(RcppCNPy)
inactivated_filter_index_target <- npyLoad(paste(folder, '/inactivated_filter_index.npy', sep = ""), "integer")

filter_num <- list()
for (i in 1:300){
  filter_num[[i]] <- i - 1
}
filter_num[inactivated_filter_index_target+1] <- NULL

for (i in 1:dim(memeMatrix)[1]){
  m <- memeMatrix[i,1:4,1:19]
  p <- makePWM(m)
  save_file_name = paste("filter", filter_num[i], "_logo.png", sep = "")
  save_name = paste(output_folder, save_file_name, sep = "")
  png(save_name)
  seqLogo(p)
  dev.off()
}


###################################################
### slots
###################################################
# slotNames(p)
# p@pwm
# p@ic
# p@consensus


###################################################
### cosmoArgs
###################################################
# args(seqLogo)


###################################################
### seqLogo1
###################################################
# seqLogo(p)

###################################################
### seqLogo2
###################################################
# seqLogo(p, ic.scale=FALSE)