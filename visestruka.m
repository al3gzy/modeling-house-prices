pkg load statistics
pkg load io

data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
x1 = cell2mat(data(2:end, 81));  % SalePrice
x2 = cell2mat(data(2:end, 47));  % GrLivArea
x3 = cell2mat(data(2:end, 18));  % OverallQual
x4 = cell2mat(data(2:end, 63));  % GarageArea
x5 = cell2mat(data(2:end, 20));  % YearBuilt
x6 = cell2mat(data(2:end, 55));  % TotRmsAbvGrd
x7 = cell2mat(data(2:end, 50));  % FullBath
x8 = cell2mat(data(2:end, 5));   % LotArea
x9 = cell2mat(data(2:end, 57)); % MasVnrArea

Y = x1;
X = [x2, x3, x4, x5, x6, x7, x8, x9];

valid_idx = all(~isnan(X), 2) & ~isnan(Y);
X = X(valid_idx, :);
Y = Y(valid_idx);

X_full = [ones(size(X,1),1), X];
n = size(X_full,1);
k = size(X_full,2) - 1;

% Pun model - procena koeficijenata beta
beta_full = (X_full' * X_full) \ (X_full' * Y);

% Predviđene vrednosti i rezidualne sume kvadrata
Y_hat_full = X_full * beta_full;
Y_mean = mean(Y);
S_R = sum((Y_hat_full - Y_mean).^2);
S_E = sum((Y - Y_hat_full).^2);
S_T = sum((Y - Y_mean).^2);

% F-test za pun model
F_stat_full = (S_R / k) / (S_E / (n - (k + 1)));
F_crit = finv(0.95, k, n - (k + 1));

fprintf('Pun model:\n');
fprintf('y = %.2f', beta_full(1));
for i = 1:k
    fprintf(' + %.2f*x%d', beta_full(i+1), i);
end
fprintf('\nF-statistika: %.4f\n', F_stat_full);
fprintf('F-kvantil (kriticna vrednost): %.4f\n', F_crit);
if F_stat_full > F_crit
    fprintf('Zakljucak: beta su statisticki znacajni.\n\n');
else
    fprintf('Zakljucak: beta nisu statisticki znacajni.\n\n');
end

R2_full = 1 - (S_E / S_T);
fprintf('R^2 za pun model: %.4f\n', R2_full);

% redukovani model
[beta_redukovan, kolone, z_score] = backwards_stepwise(X_full, Y);
fprintf('Beta koeficijenti:\n');
disp(beta_redukovan');
fprintf('Z-score:\n');
disp(z_score');

X_redukovan = X_full(:,kolone);
beta_redukovan = (X_redukovan' * X_redukovan) \ (X_redukovan' * Y);
Y_hat_redukovan = X_redukovan * beta_redukovan;

fprintf('Beta koeficijenti za redukovani model:\n');
disp(beta_redukovan');

% test značajnosti dodatnih prediktora
S_R_full = beta_full' * X_full' * Y;
S_R_redukovan = beta_redukovan' * X_redukovan' * Y;
S_b = S_R_full - S_R_redukovan;
h = size(X_full,2) - size(X_redukovan,2);

F_stat_redukovan = (S_b / h) / (S_E / (n - (k + 1)));
F_crit_redukovan = finv(0.95, h, n - (k + 1));

fprintf('Redukovani model:\n');
fprintf('y = %.2f', beta_redukovan(1));
for i = 1:length(beta_redukovan)-1
    fprintf(' + %.2f*x%d', beta_redukovan(i+1), i);
end
fprintf('\nF-statistika za test dodatnih prediktora: %.4f\n', F_stat_redukovan);
fprintf('F-kvantil (kriticna vrednost): %.4f\n', F_crit_redukovan);
if F_stat_redukovan > F_crit_redukovan
    fprintf('Zakljucak: dodatni prediktori JESU statisticki znacajni.\n');
else
    fprintf('Zakljucak: dodatni prediktori NISU statisticki znacajni.\n');
end

S_E_redukovan = sum((Y - Y_hat_redukovan).^2);
R2_redukovan = 1 - (S_E_redukovan / S_T);

fprintf('R^2 za redukovani model: %.4f\n', R2_redukovan);
