args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(survival))
for (f in args) {
  d <- read.csv(f)
  fit <- coxph(Surv(e, x, ev) ~ cmp + male + edu + smoke_ever, data=d, ties="efron")
  s <- summary(fit)
  cat(sprintf("%s  n=%d  deaths=%d  HR=%.4f (%.3f-%.3f)\n", basename(f), nrow(d),
              sum(d$ev), s$conf.int[1,1], s$conf.int[1,3], s$conf.int[1,4]))
  fit0 <- coxph(Surv(e, x, ev) ~ cmp, data=d, ties="efron")
  s0 <- summary(fit0)
  cat(sprintf("    unadjusted HR=%.4f (%.3f-%.3f)\n", s0$conf.int[1,1],
              s0$conf.int[1,3], s0$conf.int[1,4]))
}
