pkg load io
pkg load statistics

data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
OverallQual = cell2mat(data(2:end, 18));
SalePrice   = cell2mat(data(2:end, 81));
valid_idx = ~isnan(OverallQual) & ~isnan(SalePrice);
OverallQual = OverallQual(valid_idx);
SalePrice = SalePrice(valid_idx);

grupe = unique(OverallQual);
k = length(grupe);
n = length(SalePrice);
Y_bar = mean(SalePrice);

SSB = 0;
SSW = 0;

for i = 1:k
    grupa_idx = (OverallQual == grupe(i));
    Y_i = SalePrice(grupa_idx);
    n_i = length(Y_i);
    Y_i_bar = mean(Y_i);

    SSB += n_i * (Y_i_bar - Y_bar)^2;
    SSW += sum((Y_i - Y_i_bar).^2);
end

SST = SSB + SSW;

dfb = k - 1;
dfw = n - k;

MSB = SSB / dfb;
MSW = SSW / dfw;
F_stat = MSB / MSW;
F_crit = finv(0.95, dfb, dfw);
p_val = 1 - fcdf(F_stat, dfb, dfw);

fprintf('Jednofaktorska ANOVA za OverallQual:\n');
fprintf('F = %.4f\n', F_stat);
fprintf('Kritična vrednost F(%.0f, %.0f) = %.4f\n', dfb, dfw, F_crit);
fprintf('p-vrednost = %.4f\n', p_val);

if F_stat > F_crit
    fprintf('Postoji statistički značajna razlika u prosečnoj ceni između grupa OverallQual.\n');
else
    fprintf('Nema statistički značajne razlike u prosečnoj ceni između grupa OverallQual.\n');
end
