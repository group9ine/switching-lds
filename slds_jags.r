library(rjags)

data = read.csv("data/nascar/dataset.csv")
names(data) = c("x","y")

#plot(data$x, data$y, type="l")

mod_data=NULL
mod_data$x=data[seq(1,length(data), 100)]
mod_data$T=length(mod_data$y)

mod <- jags.model("jags/slds.bug", mod_data)

update(mod, 1000) # burn-in

chain <- coda.samples(mod , c("l", "y"), n.iter=10000) # sample

summary(chain)
plot(chain)
