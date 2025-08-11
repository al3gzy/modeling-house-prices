% Učitaj podatke
data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
GrLivArea = cell2mat(data(2:end, 47));
OverallQual = cell2mat(data(2:end, 18));
YearBuilt = cell2mat(data(2:end, 20));
SalePrice = cell2mat(data(2:end, 81));
SalePrice = cell2mat(data(2:end, 81));

valid = ~isnan(GrLivArea) & ~isnan(SalePrice) & ~isnan(OverallQual) & ~isnan(YearBuilt);
x = [GrLivArea(valid), OverallQual(valid), YearBuilt(valid)];
y = SalePrice(valid);


function tree = build(x, y, depth, max_depth, min_samples)
    n = size(x, 1);
    d = size(x, 2);

    if depth >= max_depth || n <= min_samples
        tree.is_leaf = true;
        tree.prediction = mean(y);
        return;
    end

    best_mse = inf;
    best_split.feature = -1;
    best_split.threshold = -1;

    for feature = 1:d
        x_feat = x(:, feature);
        [x_sorted, idx] = sort(x_feat);

        for i = 1:(n - 1)
            threshold = (x_sorted(i) + x_sorted(i+1)) / 2;
            left_idx = x_feat <= threshold;
            right_idx = x_feat > threshold;

            if sum(left_idx) < min_samples || sum(right_idx) < min_samples
                continue;
            end

            y_left = y(left_idx);
            y_right = y(right_idx);

            mse_left = mean((y_left - mean(y_left)).^2);
            mse_right = mean((y_right - mean(y_right)).^2);
            total_mse = (length(y_left)*mse_left + length(y_right)*mse_right) / n;

            if total_mse < best_mse
                best_mse = total_mse;
                best_split.feature = feature;
                best_split.threshold = threshold;
            end
        end
    end

    if best_split.feature == -1
        tree.is_leaf = true;
        tree.prediction = mean(y);
        return;
    end

    tree.is_leaf = false;
    tree.feature = best_split.feature;
    tree.threshold = best_split.threshold;

    left_idx = x(:, tree.feature) <= tree.threshold;
    right_idx = x(:, tree.feature) > tree.threshold;

    tree.left = build(x(left_idx, :), y(left_idx), depth + 1, max_depth, min_samples);
    tree.right = build(x(right_idx, :), y(right_idx), depth + 1, max_depth, min_samples);
end

function y_pred = predict(tree, x)
    y_pred = zeros(size(x, 1), 1);
    for i = 1:size(x, 1)
        y_pred(i) = traverse(tree, x(i, :));
    end
end

function y = traverse(tree, xi)
    if tree.is_leaf
        y = tree.prediction;
    elseif xi(tree.feature) <= tree.threshold
        y = traverse(tree.left, xi);
    else
        y = traverse(tree.right, xi);
    end
end

% parametri
max_depth = 8;
min_samples = 15;

% train i predikcija
tree = build(x, y, 0, max_depth, min_samples);
y_pred = predict(tree, x);

% materike
mse = mean((y - y_pred).^2);
r2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);

fprintf('MSE: %.2f\n', mse);
fprintf('R^2: %.4f\n', r2);

% vizualizacija
scatter(x(:, 1), y, 'b.');
hold on;
[sorted, idx] = sort(x(:, 1));
plot(sorted, y_pred(idx), 'r-', 'LineWidth', 2);
xlabel('GrLivArea');
ylabel('SalePrice');
title('Decision Tree sa više atributa');
legend('Podaci', 'Predikcija');
grid on;
