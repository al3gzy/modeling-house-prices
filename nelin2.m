pkg load statistics
pkg load io


data = csv2cell('/Users/aleksamilovanovic/Downloads/house-prices-advanced-regression-techniques/train.csv');
GrLivArea = cell2mat(data(2:end, 47)); % GrLivArea
SalePrice = cell2mat(data(2:end, 81)); % SalePrice

valid_idx = ~isnan(GrLivArea) & ~isnan(SalePrice);
x = GrLivArea(valid_idx);
y = SalePrice(valid_idx);

p = [0.01; 90; 20000; 0.3; 0.02; 0]; % [a, b, c, alpha, omega, phi]

f_parabola_sinus = @(p,x) ...
    (p(1)*x.^2 + p(2)*x + p(3)) + ...
    p(4)*(p(1)*x.^2 + p(2)*x + p(3)).*sin(p(5)*x + p(6));

residuals = @(p,x,y) y - f_parabola_sinus(p,x);

jacobian = @(p,x) [ ...
    -(x.^2 + p(4)*x.^2.*sin(p(5)*x + p(6))), ...
    -(x + p(4)*x.*sin(p(5)*x + p(6))), ...
    -(1 + p(4)*sin(p(5)*x + p(6))), ...
    -((p(1)*x.^2 + p(2)*x + p(3)).*sin(p(5)*x + p(6))), ...
    -p(4)*(p(1)*x.^2 + p(2)*x + p(3)).*x.*cos(p(5)*x + p(6)), ...
    -p(4)*(p(1)*x.^2 + p(2)*x + p(3)).*cos(p(5)*x + p(6)) ...
    ];

% gaus-njutn iteracije
tol = 1e-6;
max_iter = 100;
for iter = 1:max_iter
    r = residuals(p, x, y);
    J = jacobian(p, x);
    delta = pinv(J'*J) * (J'*r);
    p_new = p - delta;
    if norm(p_new - p) < tol
        break;
    end
    p = p_new;
end

% metrike
y_pred = f_parabola_sinus(p, x);
mse = mean((y - y_pred).^2);
r2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);

fprintf('MSE fitovane parabole uz oscilacije: %.2f\n', mse);
fprintf('R^2 fitovane parabole uz oscilacije: %.4f\n', r2);
fprintf('a = %.5f\n', p(1));
fprintf('b = %.5f\n', p(2));
fprintf('c = %.5f\n', p(3));
fprintf('alpha = %.5f\n', p(4));
fprintf('omega = %.5f\n', p(5));
fprintf('phi = %.5f\n', p(6));

% crtanje
x_seq = linspace(min(x), max(x), 1000);
y_init = (0.01*x_seq.^2 + 90*x_seq + 20000) + 0.3*(0.01*x_seq.^2 + 90*x_seq + 20000).*sin(0.02*x_seq);
y_fit = f_parabola_sinus(p, x_seq);

figure;
scatter(x, y, 'b.');
hold on;
plot(x_seq, y_init, 'r--', 'LineWidth', 1);
plot(x_seq, y_fit, 'k-', 'LineWidth', 2);
xlabel('GrLivArea');
ylabel('SalePrice');
title('Inicijalna i fitovana parabola uz oscilacije');
legend('Podaci', 'Inicijalna parabola sa oscilacijama', 'Fitovana parabola sa oscilacijama');
grid on;
hold off;
