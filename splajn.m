pkg load io;
pkg load statistics;

data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
GrLivArea = cell2mat(data(2:end, 47));
SalePrice = cell2mat(data(2:end, 81));

valid_idx = ~isnan(GrLivArea) & ~isnan(SalePrice);
x_all = GrLivArea(valid_idx);
y_all = SalePrice(valid_idx);

[x_sorted, sort_idx] = sort(x_all);
y_sorted = y_all(sort_idx);

num_outliers_to_remove = 5;
x_sorted = x_sorted(1:end - num_outliers_to_remove);
y_sorted = y_sorted(1:end - num_outliers_to_remove);

% čvorovi
step = 140;
indices = 1:step:length(x_sorted);
if indices(end) ~= length(x_sorted)
    indices = [indices, length(x_sorted)];
end
x_nodes = x_sorted(indices);
y_nodes = y_sorted(indices);
n = length(x_nodes);

% kubni splajn
h = diff(x_nodes);
b = diff(y_nodes) ./ h;

A = zeros(n);
rhs = zeros(n, 1);
A(1, 1) = 1;
A(end, end) = 1;
for i = 2:n-1
    A(i, i-1) = h(i-1);
    A(i, i)   = 2 * (h(i-1) + h(i));
    A(i, i+1) = h(i);
    rhs(i) = 3 * (b(i) - b(i-1));
end

m = A \ rhs;

% koeficijenti
a = y_nodes(1:end-1);
c = m(1:end-1);
d = (m(2:end) - m(1:end-1)) ./ (3 * h);
b = b - h .* (2 * m(1:end-1) + m(2:end)) / 3;

spline_eval = @(xq) arrayfun(@(xqi) ...
    eval_spline_segment(xqi, x_nodes, a, b, c, d), xq);

function y = eval_spline_segment(xq, x_nodes, a, b, c, d)
    i = find(xq >= x_nodes(1:end-1) & xq <= x_nodes(2:end), 1, 'last');
    if isempty(i)
        if xq < x_nodes(1)
            i = 1;
        else
            i = length(x_nodes) - 1;
        end
    end
    dx = xq - x_nodes(i);
    y = a(i) + b(i)*dx + c(i)*dx^2 + d(i)*dx^3;
end

% linearni produžetak iz poslednje tačke
last_x = x_nodes(end);
last_y = y_nodes(end);
last_slope = b(end);

linear_extension = @(xq) last_y + last_slope * (xq - last_x);

% crtanje
xq = linspace(min(x_sorted), max(x_sorted) + 200, 1000);
yq = zeros(size(xq));
for i = 1:length(xq)
    if xq(i) <= last_x
        yq(i) = spline_eval(xq(i));
    else
        yq(i) = linear_extension(xq(i));
    end
end

figure;
scatter(x_all, y_all, 10, 'b', 'filled'); hold on;
plot(xq, yq, 'r-', 'LineWidth', 2);
xlabel('GrLivArea');
ylabel('SalePrice');
title('Splajn interpolacija sa linearnim nastavkom');
legend('Podaci', 'Splajn + linearna ekstrapolacija');
grid on;

% metrike (samo u validnom domenu splajna)
in_domain_idx = x_sorted <= last_x;
x_in = x_sorted(in_domain_idx);
y_in = y_sorted(in_domain_idx);
y_pred_in = spline_eval(x_in);

mse_interp = mean((y_in - y_pred_in).^2);
r2_interp = 1 - sum((y_in - y_pred_in).^2) / sum((y_in - mean(y_in)).^2);

fprintf('MSE: %.2f\n', mse_interp);
fprintf('R^2: %.4f\n', r2_interp);