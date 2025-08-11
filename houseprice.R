# učitavamo biblioteke
library(tidyverse)
library(ggplot2)
library(corrplot)
library(dplyr)
library(lmtest)      
library(tibble)

# učitavamo podatke
data <- read.csv("/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv")

# histogram cene kuće
ggplot(data, aes(x = SalePrice)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "black") +
  labs(title = "Distribucija cena kuća", x = "Cena", y = "Frekvencija")

# histogram kvadrature
ggplot(data, aes(x = GrLivArea)) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "black") +
  labs(title = "Distribucija kvadrature kuće", x = "GrLivArea", y = "Frekvencija")

# izbor numeričkih promenljivih
numeric_vars <- select(data, where(is.numeric))
cor_matrix <- cor(numeric_vars, use = "pairwise.complete.obs")

# heatmap
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6)

# scatterplot
ggplot(data, aes(x = GrLivArea, y = LotArea)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Odnos kvadrature i cene",
       x = "GrLivArea (kvadratne stope)",
       y = "SalePrice (USD)")

# jednostruka regresija ----

# zadržavamo samo redove bez NA vrednosti za GrLivArea i SalePrice
data <- na.omit(data[, c("GrLivArea", "SalePrice")])

# podela na trening i test skup
set.seed(123)
n <- nrow(data)
train_idx <- sample(1:n, size = 0.8 * n)
train_set <- data[train_idx, ]
test_set <- data[-train_idx, ]

# metod najmanjih kvadrata
X <- train_set$GrLivArea
Y <- train_set$SalePrice
n_train <- length(Y)

beta_hat <- sum((X - mean(X)) * (Y - mean(Y))) / sum((X - mean(X))^2)
alpha_hat <- mean(Y) - beta_hat * mean(X)

Y_hat <- alpha_hat + beta_hat * X
residuals <- Y - Y_hat

# procena varijanse i standardna greška za beta
sigma_hat_sq <- sum(residuals^2) / (n_train - 2)
SE_beta <- sqrt(sigma_hat_sq / sum((X - mean(X))^2))

# t-test za beta
t_beta <- beta_hat / SE_beta
p_value <- 2 * pt(-abs(t_beta), df = n_train - 2)

# interval poverenja za beta
alpha <- 0.05
t_crit <- qt(1 - alpha/2, df = n_train - 2)
CI_lower <- beta_hat - t_crit * SE_beta
CI_upper <- beta_hat + t_crit * SE_beta

# koeficijent determinacije na trening skupu
SS_tot <- sum((Y - mean(Y))^2)
SS_res <- sum(residuals^2)
R2 <- 1 - SS_res / SS_tot

# rezultati na trening skupu
cat("Trening skup\n")
cat("Alfa:", alpha_hat, "\n")
cat("Beta:", beta_hat, "\n")
cat("SE(beta):", SE_beta, "\n")
cat("t-statistika:", t_beta, "\n")
cat("p-vrednost:", p_value, "\n")
cat("95% interval poverenja za beta: (", CI_lower, ",", CI_upper, ")\n")
cat("R^2:", R2, "\n\n")

# qq plot reziduala
qqnorm(residuals)
qqline(residuals, col = "red")

# reziduali i predikcije
plot(Y_hat, residuals,
     xlab = "Predviđene vrednosti",
     ylab = "Reziduali",
     main = "Reziduali vs Predikcije (trening)")
abline(h = 0, col = "red")

# testiranje modela na test skupu
X_test <- test_set$GrLivArea
Y_test <- test_set$SalePrice
Y_pred_test <- alpha_hat + beta_hat * X_test
residuals_test <- Y_test - Y_pred_test

# MSE i R2 na test skupu
MSE_test <- mean(residuals_test^2)
SS_tot_test <- sum((Y_test - mean(Y_test))^2)
SS_res_test <- sum(residuals_test^2)
R2_test <- 1 - SS_res_test / SS_tot_test

# rezultati na test skupu
cat("Test skup\n")
cat("MSE:", MSE_test, "\n")
cat("R^2:", R2_test, "\n")

# interval poverenja za E(Y0) i Y0 pri zadatom x0
x0 <- 1860  
Y0_hat <- alpha_hat + beta_hat * x0

# interval poverenja za očekivanu vrednost E(Y0)
SE_mean <- sqrt(sigma_hat_sq * (1 / n_train + (x0 - mean(X))^2 / sum((X - mean(X))^2)))
CI_mean_lower <- Y0_hat - t_crit * SE_mean
CI_mean_upper <- Y0_hat + t_crit * SE_mean

cat("Interval poverenja za E(Y0) pri x0 =", x0, ": (", CI_mean_lower, ",", CI_mean_upper, ")\n")

# interval poverenja za posmatranu vrednost Y0
SE_pred <- sqrt(sigma_hat_sq * (1 + 1 / n_train + (x0 - mean(X))^2 / sum((X - mean(X))^2)))
CI_pred_lower <- Y0_hat - t_crit * SE_pred
CI_pred_upper <- Y0_hat + t_crit * SE_pred

cat("Interval poverenja za Y0 pri x0 =", x0, ": (", CI_pred_lower, ",", CI_pred_upper, ")\n")

# testiranje H0: ro = 0 (korelacija) 
t_ro <- beta_hat / SE_beta  # ista t-statistika kao i za betu
cat("t-statistika za H0: ro = 0 je", t_ro, "\n")
if (abs(t_ro) > t_crit) {
  cat("ro je statistički značajan.\n")
} else {
  cat("ro nije statistički značajan.\n")
}

# višestruka regresija -----

# pun model
data <- read.csv("/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv")
model_data <- data[, c("SalePrice", "GrLivArea", "OverallQual", "GarageArea", "YearBuilt", 
                       "TotRmsAbvGrd", "FullBath", "LotArea", "X1stFlrSF", 
                       "MasVnrArea", "Fireplaces")]
model_data <- na.omit(model_data)

Y <- as.matrix(model_data$SalePrice)
X_full <- as.matrix(cbind(Intercept = 1, model_data[, -1]))

n <- nrow(X_full)
k <- ncol(X_full) - 1  # bez intercepta

beta_full <- solve(t(X_full) %*% X_full) %*% t(X_full) %*% Y
Y_hat_full <- X_full %*% beta_full
Y_mean <- mean(Y)
S_R <- sum((Y_hat_full - Y_mean)^2)
S_E <- sum((Y - Y_hat_full)^2)

# f-test za pun model 
F_stat_full <- (S_R / k) / (S_E / (n - (k + 1)))
F_crit <- qf(0.95, df1 = k, df2 = n - (k + 1))

cat("Pun model:\n")
cat("y =", round(beta_full[1], 2), "+", 
    paste0(round(beta_full[-1], 2), "*x", 1:k, collapse = " + "), "\n")
cat("F-statistika:", round(F_stat_full, 4), "\n")
cat("F-kvantil (kritična vrednost):", round(F_crit, 4), "\n")
if (F_stat_full > F_crit) {
  cat("Zaključak: beta su statistički značajni.\n\n")
} else {
  cat("Zaključak: beta nisu statistički značajni.\n\n")
}

# redukovani model 

rss <- function(X, y, beta) {
  sum((y - X %*% beta)^2)
}

backward_stepwise <- function(X, y) {
  X_trenutno <- X
  n <- nrow(X_trenutno)
  d <- ncol(X_trenutno)
  
  beta_full <- solve(t(X_trenutno) %*% X_trenutno) %*% t(X_trenutno) %*% y
  rss_stari <- rss(X_trenutno, y, beta_full)
  sigma2 <- rss_stari / (n - d)
  
  var_beta <- diag(solve(t(X_trenutno) %*% X_trenutno) * sigma2)
  z_score <- beta_full / sqrt(var_beta)
  
  izabrane_kolone <- 1:d
  izbacene_kolone <- c()
  
  F_score <- 0
  kriticna_vrednost <- 1
  
  while (F_score < kriticna_vrednost && length(izabrane_kolone) > 1) {
    indeks_najslabije <- which.min(abs(z_score))
    
    poslednje_izbacena <- izabrane_kolone[indeks_najslabije]
    izbacene_kolone <- c(izbacene_kolone, poslednje_izbacena)
    
    X_trenutno <- X_trenutno[, -indeks_najslabije, drop = FALSE]
    izabrane_kolone <- izabrane_kolone[-indeks_najslabije]
    
    beta_novi <- solve(t(X_trenutno) %*% X_trenutno) %*% t(X_trenutno) %*% y
    rss_novi <- rss(X_trenutno, y, beta_novi)
    
    sigma2 <- rss_novi / (n - ncol(X_trenutno))
    var_beta <- diag(solve(t(X_trenutno) %*% X_trenutno) * sigma2)
    z_score <- beta_novi / sqrt(var_beta)
    
    broj_izbacenih <- length(izbacene_kolone)
    F_score <- ((rss_novi - rss_stari) / broj_izbacenih) / (rss_stari / (n - d))
    kriticna_vrednost <- qchisq(0.95, df = broj_izbacenih) / broj_izbacenih
  }
  
  izabrane_kolone <- c(izabrane_kolone, poslednje_izbacena)
  izabrane_kolone <- sort(izabrane_kolone)
  
  X_konacno <- X[, izabrane_kolone, drop = FALSE]
  beta_konacno <- solve(t(X_konacno) %*% X_konacno) %*% t(X_konacno) %*% y
  
  sigma2 <- sum((y - X_konacno %*% beta_konacno)^2) / (n - ncol(X_konacno))
  var_beta <- diag(solve(t(X_konacno) %*% X_konacno) * sigma2)
  z_score <- beta_konacno / sqrt(var_beta)
  
  list(beta = beta_konacno, kolone = izabrane_kolone, z_score = z_score)
}

rezultat <- backward_stepwise(X_full, Y)

cat("Izabrane kolone:\n")
print(rezultat$kolone)
cat("Beta koeficijenti:\n")
print(rezultat$beta)
cat("Z-score:\n")
print(rezultat$z_score)

X_reduced <- X_full[, rezultat$kolone, drop = FALSE]

beta_reduced <- solve(t(X_reduced) %*% X_reduced) %*% t(X_reduced) %*% Y
Y_hat_reduced <- X_reduced %*% beta_reduced

cat("Beta koeficijenti za redukovani model:\n")
print(beta_reduced)

# test značajnosti dodatnih prediktora
S_R_full <- t(beta_full) %*% t(X_full) %*% Y
S_R_reduced <- t(beta_reduced) %*% t(X_reduced) %*% Y
S_b <- S_R_full - S_R_reduced
h <- ncol(X_full) - ncol(X_reduced)

F_stat_reduced <- (S_b / h) / (S_E / (n - (k + 1)))
F_crit_reduced <- qf(0.95, df1 = h, df2 = n - (k + 1))

cat("Redukovani model:\n")
cat("y =", round(beta_reduced[1], 2), "+", 
    paste0(round(beta_reduced[-1], 2), "*x", 1:(ncol(X_reduced) - 1), collapse = " + "), "\n")
cat("F-statistika za test dodatnih prediktora:", round(F_stat_reduced, 4), "\n")
cat("F-kvantil (kritična vrednost):", round(F_crit_reduced, 4), "\n")
if (F_stat_reduced > F_crit_reduced) {
  cat("Zaključak: dodatni prediktori JESU statistički značajni.\n")
} else {
  cat("Zaključak: dodatni prediktori NISU statistički značajni.\n")
}

# nelinearni model ---- 1
a <- 0.01
b <- 90
c <- 20000

f_parabola <- function(x, p) {
  p[1] * x^2 + p[2] * x + p[3]
}

residuals_parabola <- function(p, x, y) {
  y - f_parabola(x, p)
}

jacobian_parabola <- function(p, x) {
  cbind(
    -x^2,
    -x,
    -1
  )
}

gaus_njutn <- function(p0, x, y, residuals_fn, jacobian_fn, tol = 1e-6, max_iter = 100) {
  p <- p0
  for (i in 1:max_iter) {
    r <- residuals_fn(p, x, y)
    J <- jacobian_fn(p, x)
    delta <- solve(t(J) %*% J) %*% (t(J) %*% r)
    p_new <- p - as.vector(delta)
    if (sqrt(sum((p_new - p)^2)) < tol) break
    p <- p_new
  }
  p
}

p0 <- c(a, b, c)
fit <- gaus_njutn(p0, data$GrLivArea, data$SalePrice, residuals_parabola, jacobian_parabola)

pred_fit <- f_parabola(data$GrLivArea, fit)
mse <- mean((data$SalePrice - pred_fit)^2)
r2 <- 1 - sum((data$SalePrice - pred_fit)^2) / sum((data$SalePrice - mean(data$SalePrice))^2)

x_seq <- seq(min(data$GrLivArea), max(data$GrLivArea), length.out = 1000)

df_plot <- data.frame(
  GrLivArea = x_seq,
  Parabola_init = f_parabola(x_seq, p0),
  Parabola_fit = f_parabola(x_seq, fit)
)

ggplot(data, aes(x = GrLivArea, y = SalePrice)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_line(data = df_plot, aes(x = GrLivArea, y = Parabola_init), color = "orange", linetype = "dashed", size = 1) +
  geom_line(data = df_plot, aes(x = GrLivArea, y = Parabola_fit), color = "red", size = 1.2) +
  labs(
    title = paste0("Inicijalna i fitovana parabola"),
    x = "GrLivArea",
    y = "SalePrice"
  )

cat("MSE fitovane parabole:", round(mse, 2), "\n")
cat("R² fitovane parabole:", round(r2, 4), "\n")
cat("a =", round(fit[1], 5), "\n")
cat("b =", round(fit[2], 5), "\n")
cat("c =", round(fit[3], 5), "\n\n")

# nelinearni model ---- 2
a <- 0.01
b <- 90
c <- 20000
alpha <- 0.3
omega <- 0.02
phi <- 0

f_parabola_sinus <- function(x, p) {
  a <- p[1]; b <- p[2]; c <- p[3]
  alpha <- p[4]; omega <- p[5]; phi <- p[6]
  parabola <- a * x^2 + b * x + c
  parabola + alpha * parabola * sin(omega * x + phi)
}

residuals_parabola_sinus <- function(p, x, y) {
  y - f_parabola_sinus(x, p)
}

jacobian_parabola_sinus <- function(p, x) {
  a <- p[1]; b <- p[2]; c <- p[3]
  alpha <- p[4]; omega <- p[5]; phi <- p[6]
  parabola <- a * x^2 + b * x + c
  sin_part <- sin(omega * x + phi)
  cos_part <- cos(omega * x + phi)
  
  cbind(
    - (x^2 + alpha * x^2 * sin_part),
    - (x + alpha * x * sin_part),
    - (1 + alpha * sin_part),
    - parabola * sin_part,
    - alpha * parabola * x * cos_part,
    - alpha * parabola * cos_part
  )
}

library(MASS)

gaus_njutn1 <- function(p0, x, y, residuals_fn, jacobian_fn, tol = 1e-6, max_iter = 100) {
  p <- p0
  for (i in 1:max_iter) {
    r <- residuals_fn(p, x, y)
    J <- jacobian_fn(p, x)
    
    delta <- ginv(t(J) %*% J) %*% (t(J) %*% r)
    
    p_new <- p - as.vector(delta)
    
    if (sqrt(sum((p_new - p)^2)) < tol) break
    p <- p_new
  }
  p
}

p0 <- c(a, b, c, alpha, omega, phi)
fit <- gaus_njutn1(p0, data$GrLivArea, data$SalePrice, residuals_parabola_sinus, jacobian_parabola_sinus)

pred_fit <- f_parabola_sinus(data$GrLivArea, fit)
mse <- mean((data$SalePrice - pred_fit)^2)
r2 <- 1 - sum((data$SalePrice - pred_fit)^2) / sum((data$SalePrice - mean(data$SalePrice))^2)

x_seq <- seq(min(data$GrLivArea), max(data$GrLivArea), length.out = 1000)

df_plot <- data.frame(
  GrLivArea = x_seq,
  Parabola_sin_init = f_parabola_sinus(x_seq, p0),
  Parabola_sin_fit = f_parabola_sinus(x_seq, fit)
)

ggplot(data, aes(x = GrLivArea, y = SalePrice)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_line(data = df_plot, aes(y = Parabola_sin_init), color = "orange", linetype = "dashed", size = 1) +
  geom_line(data = df_plot, aes(y = Parabola_sin_fit), color = "red", size = 1.2) +
  labs(
    title = paste0("Inicijalna i fitovana parabola uz oscilacije"),
    x = "GrLivArea",
    y = "SalePrice"
  )

cat("MSE fitovane parabole uz oscilacije:", round(mse, 2), "\n")
cat("R² fitovane parabole uz oscilacije:", round(r2, 4), "\n")
cat("a =", round(fit[1], 5), "\n")
cat("b =", round(fit[2], 5), "\n")
cat("c =", round(fit[3], 5), "\n\n")
               
# anova ----

# jednofaktorska
aov_model_1 <- aov(SalePrice ~ Neighborhood, data = data)
summary(aov_model_1)

# dvofaktorska bez interacije
aov_model_2 <- aov(SalePrice ~ Neighborhood + OverallQual, data = data)
summary(aov_model_2)

# dvofaktorska sa interakcijom
aov_model_3 <- aov(SalePrice ~ Neighborhood * OverallQual, data = data)
summary(aov_model_3)

# interakcioni dijagram
interaction.plot(data$OverallQual, data$Neighborhood, data$SalePrice)
