#R运算性能比较
library(compiler)
library(parallel)
library(microbenchmark)

n <- 100
m <- 1e5

#for循环
fun0 <- function(n,m){
  coef.sim <- matrix(0, ncol = 2, nrow = m)
  x <- runif(n, 0, 100)
  for(i in 1:m){
    e <- rnorm(n, 0, 5)
    y <- 20 + 1.5*x + e
    coef.sim[i, 1] <- cov(x,y)/var(x)
    coef.sim[i, 2] <- (y[2] - y[1])/(x[2] - x[1])
  }
  return(list(Means = colMeans(coef.sim), SD = apply(coef.sim, 2, sd)))
}
system.time(fun0(n,m))

fun1 <- function(n,m){
  coef.sim <- matrix(0, ncol = 2, nrow = m)
  x <- runif(n, 0, 100)
  for(i in 1:m){
    e <- rnorm(n, 0, 5)
    y <- 20 + 1.5*x + e
    coef.sim[i, 1] <- lm(y~x)$coefficients[2]
    coef.sim[i, 2] <- (y[2] - y[1])/(x[2] - x[1])
  }
  return(list(Means = colMeans(coef.sim), SD = apply(coef.sim, 2, sd)))
}

system.time(fun1(n,m))

#向量化
fun2 <- function(n,m){
  x <- runif(n,0,100)
  e <- lapply(rep(n, m), rnorm, mean = 0, sd = 5)
  y <- lapply(e,function(x,e){20 + 1.5*x +e},x = x)
  coef.sim <- lapply(y, function(y,x){
    c(cov(x,y)/var(x),(y[2]-y[1])/(x[2] - x[1]))
  }, x = x)
  coef.sim <- do.call('rbind',coef.sim)
  return(list(Means = colMeans(coef.sim), SD = apply(coef.sim, 2, sd)))
}

fun2_0 <- function(n,m){
  x <- runif(n,0,100)
  e <- lapply(rep(n, m), rnorm, mean = 0, sd = 5)
  coef.sim <- lapply(e, function(x,e){
    y = 20 + 1.5*x + e
    c(cov(x,y)/var(x),(y[2]-y[1])/(x[2] - x[1]))
    }, x = x)
  coef.sim <- do.call('rbind',coef.sim)
  return(list(Means = rowMeans(coef.sim), SD = apply(coef.sim, 1, sd)))
}

system.time(fun2(n,m))
system.time(fun2_0(n,m))


#字节码编译
fun3 <- cmpfun(fun0)
system.time(fun3(n,m))

fun4 <- cmpfun(fun2)
system.time(fun4(n,m))
#利用CPP

#并行计算-CPU
tic <- Sys.time()
x <- runif(n, 0, 100)
e <- lapply(rep(n, m), rnorm, mean = 0, sd = 5)
nc <- detectCores()
cl <- makeCluster(nc - 1)
clusterExport(cl,"x")
coef.sim <- parLapply(cl,e,function(e){
  y <- 20 + 1.5*x + e
  c(cov(x,y)/var(x),(y[2]-y[1])/(x[2] - x[1]))
})
coef.sim <- do.call('rbind',coef.sim)
stopCluster(cl)
list(Means = colMeans(coef.sim), SD = apply(coef.sim, 2, sd))
Sys.time() - tic

#parallel - GPU
library(gpuR)
detectGPUs()
gpuInfo()
len <- 1000

x <- matrix(runif(len*len) , ncol = len)
system.time(y <- crossprod(x,x))
system.time(z <- eigen(y, only.values = T))

x1 <- gpuMatrix(x, type = "float")
system.time(y1 <- crossprod(x1,x1))
system.time(z1 <- eigen(y1,symmetric = T, only.values = T))

#操作失败，gpuR无法识别gpu,后来在一台台式机上试验成功。速度提升很多！
########################################################
