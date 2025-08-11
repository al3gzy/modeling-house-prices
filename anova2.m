pkg load io
pkg load statistics

data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
OverallQual = cell2mat(data(2:end, 18));
YearBuilt = cell2mat(data(2:end, 20));
SalePrice = cell2mat(data(2:end, 81));
[uniqueQual, ~, qualIdx] = unique(OverallQual);
[uniqueYear, ~, yearIdx] = unique(YearBuilt);

numQual = length(uniqueQual);
numYear = length(uniqueYear);
N = length(SalePrice);
grandMean = mean(SalePrice);

meanQual = zeros(numQual,1);
for i = 1:numQual
  meanQual(i) = mean(SalePrice(qualIdx == i));
end

meanYear = zeros(numYear,1);
for j = 1:numYear
  meanYear(j) = mean(SalePrice(yearIdx == j));
end

meanGroup = zeros(numQual,numYear);
for i = 1:numQual
  for j = 1:numYear
    meanGroup(i,j) = mean(SalePrice(qualIdx == i & yearIdx == j));
  end
end

SST = sum((SalePrice - grandMean).^2);

SSA = 0;
for i = 1:numQual
  n_i = sum(qualIdx == i);
  SSA = SSA + n_i * (meanQual(i) - grandMean)^2;
end

SSB = 0;
for j = 1:numYear
  n_j = sum(yearIdx == j);
  SSB = SSB + n_j * (meanYear(j) - grandMean)^2;
end

SSAB = 0;
for i = 1:numQual
  for j = 1:numYear
    n_ij = sum((qualIdx == i) & (yearIdx == j));
    if n_ij > 0
      SSAB = SSAB + n_ij * (meanGroup(i,j) - meanQual(i) - meanYear(j) + grandMean)^2;
    end
  end
end

SSE = 0;
for k = 1:N
  i = qualIdx(k);
  j = yearIdx(k);
  SSE = SSE + (SalePrice(k) - meanGroup(i,j))^2;
end

% stepeni slobode
dfA = numQual - 1;
dfB = numYear - 1;
dfAB = dfA * dfB;
dfE = N - numQual * numYear;
dfT = N - 1;

% srednje kvadrate
MSA = SSA / dfA;
MSB = SSB / dfB;
MSAB = SSAB / dfAB;
MSE = SSE / dfE;

% f-statistike
FA = MSA / MSE;
FB = MSB / MSE;
FAB = MSAB / MSE;

% vizualizacija
fprintf('Dvofaktorska ANOVA za YearBuilt i OverallQual:\n');
fprintf('Faktor OverallQual: F = %.3f\n', FA);
fprintf('Faktor YearBuilt: F = %.3f\n', FB);
fprintf('Interakcija: F = %.3f\n', FAB);
fprintf('Suma kvadrata i df:\n');
fprintf('SSA = %.2f, dfA = %d\n', SSA, dfA);
fprintf('SSB = %.2f, dfB = %d\n', SSB, dfB);
fprintf('SSAB = %.2f, dfAB = %d\n', SSAB, dfAB);
fprintf('SSE = %.2f, dfE = %d\n', SSE, dfE);
fprintf('SST = %.2f, dfT = %d\n', SST, dfT);
