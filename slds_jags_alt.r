library(rjags)
library(ggplot2)

data = read.csv("data/nascar/dataset.csv")
#names(data) = c("x","y")

#plot(data$x, data$y, type="l")
D=2
K=5

sdata=data.frame(nrow=nrow(data)-1)
sdata[1:nrow(data)-1,1]=data[2:nrow(data),1]
sdata[1:nrow(data)-1,2]=data[2:nrow(data),2]


data=data[1:nrow(data)-1,]

subsamp <- seq(1,nrow(data),1)

mod_data=NULL
mod_data$x=data[subsamp,]
mod_data$sx=sdata[subsamp,]
mod_data$T=nrow(mod_data$x)
mod_data$K=K
mod_data$D=D
#mod_data$x0=mod_data$x[1,]

T=mod_data$T
niter=10


mod <- jags.model("jags/slds_alt.bug", mod_data)

update(mod, 10000) # burn-in

chain <- coda.samples(mod , c("sx","z"), n.iter=niter) # sample

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
  results.list[[i]]$x1[j]=chain.df[i,paste("x[",j,",1]",sep="")]
  results.list[[i]]$x2[j]=chain.df[i,paste("x[",j,",2]",sep="")]
  results.list[[i]]$z[j]=chain.df[i,paste("z[",j,"]",sep="")]
  }, error=function(e){print(i);print(j); print(e);})
 }
 results.list[[i]]$z = as.factor(results.list[[i]]$z)
}

pdf(file="Rplots-alt.pdf")

for (i in 1:niter) {
 plot(ggplot() + geom_point(aes(results.list[[i]]$x1, results.list[[i]]$x2, col=results.list[[i]]$z))+geom_path(aes(results.list[[i]]$x1, results.list[[i]]$x2)))
}

dev.off()
