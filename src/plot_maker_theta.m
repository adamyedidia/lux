load('dthetadt.mat');

exes = linspace(0, 33, 856);

plot(exes, thetas, 'color', 'r', 'LineWidth', 8); hold on;

xlim([0, 33]);

zerovec = zeros(1,856);
plot(exes, zerovec, 'color', 'k', 'LineWidth', 8);


set(gca, 'FontSize', 20);
set(gcf,'color','w');

