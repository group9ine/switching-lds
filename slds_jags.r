library(rjags)
library(ggplot2)

#data = read.csv("data/nascar/dataset.csv")
data = read.csv("data/dataset.csv")
names(data) = c("x","y")


truez = c(1, sign(diff(atan2(data[,2],data[,1]))))
truez[truez==-1]=2

scale = (sum(diff(as.matrix(data))**2)/(nrow(data)-1))**0.5
D=2
K=2

mod_data=NULL
mod_data$x=data[seq(1,nrow(data)/10,1),]*10/scale
# mod_data$x=data*10/scale
mod_data$T=nrow(mod_data$x)
mod_data$K=K
mod_data$D=D
# mod_data$z=truez
#mod_data$bscale=bscale
#mod_data$x0=mod_data$x[1,]



T=mod_data$T
niter=10



plot(ggplot(data) +geom_point(aes(x, y, col=as.factor(truez)))) #see what the data is

mod <- jags.model("jags/slds.bug", mod_data)

update(mod, 10000) # burn-in

chain <- coda.samples(mod , c("predx","predz"), n.iter=niter) # sample

chain.df = as.data.frame(as.mcmc(chain))

#print(chain2)

results.list=list()
for (i in 1:niter) {
 results.list[[i]]=list()
 results.list[[i]]$x1=NaN*seq(T)
 results.list[[i]]$x2=NaN*seq(T)
 results.list[[i]]$z=NaN*seq(T)
 for (j in 1:T) {
  tryCatch({
  results.list[[i]]$x1[j]=chain.df[i,paste("predx[",j,",1]",sep="")]
  results.list[[i]]$x2[j]=chain.df[i,paste("predx[",j,",2]",sep="")]
  results.list[[i]]$z[j]=chain.df[i,paste("predz[",j,"]",sep="")]
  }, error=function(e){print(i);print(j); print(e);})
 }
 results.list[[i]]$z = as.factor(results.list[[i]]$z)
}


print(coda.samples(mod, c("Ab", "Q", "pi"), n.iter=1))

for (i in 1:niter) {
 plot(ggplot() + geom_point(aes(results.list[[i]]$x1, results.list[[i]]$x2, col=results.list[[i]]$z))+geom_path(aes(results.list[[i]]$x1, results.list[[i]]$x2)))
}
print("Done")
