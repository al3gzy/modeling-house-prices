pkg load io
pkg load statistics

% učitavanje podataka
data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
X = cell2mat(data(2:end, 47));
Y = cell2mat(data(2:end, 81));

valid_idx = ~isnan(X) & ~isnan(Y);
X = X(valid_idx);
Y = Y(valid_idx);

% histogram Y
figure;
hist(Y, 50);
title('Distribucija cena kuća');
xlabel('Cena');
ylabel('Frekvencija');

% histogram X
figure;
hist(X, 50);
title('Distribucija kvadrature kuće');
xlabel('GrLivArea');
ylabel('Frekvencija');

% scatterplot X vs Y
figure;
scatter(X, Y, 'b.');
hold on;
p = polyfit(X, Y, 1);
x_vals = linspace(min(X), max(X), 100);
y_vals = polyval(p, x_vals);
plot(x_vals, y_vals, 'r-', 'LineWidth', 2);
title('Odnos kvadrature i cene');
xlabel('GrLivArea');
ylabel('SalePrice');

% podela na trening/test
rand('seed', 123);
n = length(Y);
idx = randperm(n);
train_idx = idx(1:round(0.8*n));
test_idx = idx(round(0.8*n)+1:end);

X_train = X(train_idx);
Y_train = Y(train_idx);
X_test = X(test_idx);
Y_test = Y(test_idx);

% linearna regresija
X_mean = mean(X_train);
Y_mean = mean(Y_train);
beta_hat = sum((X_train - X_mean) .* (Y_train - Y_mean)) / sum((X_train - X_mean).^2);
alpha_hat = Y_mean - beta_hat * X_mean;

Y_hat = alpha_hat + beta_hat * X_train;
residuals = Y_train - Y_hat;

% statistika
n_train = length(Y_train);
sigma2 = sum(residuals.^2) / (n_train - 2);
SE_beta = sqrt(sigma2 / sum((X_train - X_mean).^2));
t_beta = beta_hat / SE_beta;
p_val = 2 * (1 - tcdf(abs(t_beta), n_train - 2));
t_crit = tinv(0.975, n_train - 2);
CI_beta = [beta_hat - t_crit * SE_beta, beta_hat + t_crit * SE_beta];
SS_tot = sum((Y_train - Y_mean).^2);
SS_res = sum(residuals.^2);
R2 = 1 - SS_res / SS_tot;

fprintf('Trening skup:\n');
fprintf('Alfa: %.2f\n', alpha_hat);
fprintf('Beta: %.4f\n', beta_hat);
fprintf('SE(beta): %.4f\n', SE_beta);
fprintf('t-statistika: %.4f\n', t_beta);
fprintf('p-vrednost: %.4f\n', p_val);
fprintf('95%% CI za beta: [%.4f, %.4f]\n', CI_beta(1), CI_beta(2));
fprintf('R^2: %.4f\n', R2);

% QQ plot
figure;
res_sorted = sort(residuals);
norm_quantiles = norminv(((1:n_train) - 0.5) / n_train);
plot(norm_quantiles, res_sorted, 'bo');
hold on;
plot(norm_quantiles, norm_quantiles, 'r-');
title('QQ plot reziduala');

% reziduali vs predikcije
figure;
scatter(Y_hat, residuals, 'b.');
xlabel('Predviđene vrednosti');
ylabel('Reziduali');
title('Reziduali vs Predikcije (trening)');
line(xlim, [0 0], 'Color', 'r');

% test skup
Y_pred_test = alpha_hat + beta_hat * X_test;
residuals_test = Y_test - Y_pred_test;

MSE_test = mean(residuals_test.^2);
R2_test = 1 - sum(residuals_test.^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('\nTest skup:\n');
fprintf('MSE: %.4f\n', MSE_test);
fprintf('R^2: %.4f\n', R2_test);

% intervali poverenja
x0 = 1860;
Y0_hat = alpha_hat + beta_hat * x0;
SE_mean = sqrt(sigma2 * (1/n_train + (x0 - X_mean)^2 / sum((X_train - X_mean).^2)));
SE_pred = sqrt(sigma2 * (1 + 1/n_train + (x0 - X_mean)^2 / sum((X_train - X_mean).^2)));

CI_mean = [Y0_hat - t_crit * SE_mean, Y0_hat + t_crit * SE_mean];
CI_pred = [Y0_hat - t_crit * SE_pred, Y0_hat + t_crit * SE_pred];

fprintf('\nInterval poverenja za E(Y0) pri x0 = %d: [%.2f, %.2f]\n', x0, CI_mean(1), CI_mean(2));
fprintf('Interval poverenja za Y0 pri x0 = %d: [%.2f, %.2f]\n', x0, CI_pred(1), CI_pred(2));

% testiranje H0: ro = 0
fprintf('\nt-statistika za H0: ro = 0 je %.4f\n', t_beta);
if abs(t_beta) > t_crit
    fprintf('ro je statistički značajan.\n');
else
    fprintf('ro nije statistički značajan.\n');
end
