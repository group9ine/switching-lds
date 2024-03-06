library(rjags)
library(ggplot2)

data = read.csv("data/nascar/dataset.csv")
#names(data) = c("x","y")

#plot(data$x, data$y, type="l")

mod_data=NULL
mod_data$x=data[seq(1,nrow(data)/10, 100),]
mod_data$T=nrow(mod_data$x)
mod_data$K=4
mod_data$D=2

T=mod_data$T
niter=10


mod <- jags.model("jags/slds.bug", mod_data)

update(mod, 1000) # burn-in

chain <- coda.samples(mod , c("Ab", "Q", "pi"), n.iter=10000) # sample

chain.df = as.data.frame(as.mcmc(chain))
chain.list = as.list(chain.df)
chain.list$T=mod_data$T
chain.list$D=mod_data$D
chain.list$K=mod_data$K
chain.list$x0=mod_data$x[1,]
mod2 <- jags.model("jags/slds-gen.bug", chain.list)

update(mod2, 1000)

chain2 <- coda.samples(mod2 , c("x","z"), n.iter=niter)

#print(chain2)

chain2.df = as.data.frame(as.mcmc(chain2))
results.list=list()
for (i in 1:niter) {
 results.list[[i]]=list()
 results.list[[i]]$x1=NaN*seq(T)
 results.list[[i]]$x2=NaN*seq(T)
 results.list[[i]]$z=NaN*seq(T)
 for (j in 1:T) {
  tryCatch({
  results.list[[i]]$x1[j]=chain2.df[i,paste("x[",j,",1]",sep="")]
  results.list[[i]]$x2[j]=chain2.df[i,paste("x[",j,",2]",sep="")]
  results.list[[i]]$z[j]=chain2.df[i,paste("z[",j,"]",sep="")]
  }, error=function(e){print(i);print(j); print(e);})
 }
}



for (i in 1:niter) {
 plot(ggplot() + geom_point(aes(results.list[[i]]$x1, results.list[[i]]$x2, col=results.list[[i]]$z)))
}
