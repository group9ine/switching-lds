library(rjags)

data = read.csv("data/nascar/dataset.csv")
#names(data) = c("x","y")

#plot(data$x, data$y, type="l")

mod_data=NULL
mod_data$x=data[seq(1,nrow(data)/10, 100),]
mod_data$T=length(mod_data$x)
mod_data$K=4
mod_data$D=2

# #load.module("") #if the ddirch is in another module, not sure

mod <- jags.model("jags/slds.bug", mod_data)

update(mod, 1000) # burn-in

chain <- coda.samples(mod , c("z", "x"), n.iter=10000) # sample

summary(chain)
plot(chain)
