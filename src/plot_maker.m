load('corner_error.mat');

topMean1 = means(1, :) + stdevs(1,:);
bottomMean1 = means(1, :) - stdevs(1, :);

topMean2 = means(2, :) + stdevs(2,:);
bottomMean2 = means(2, :) - stdevs(2, :);

topMean3 = means(3, :) + stdevs(3,:);
bottomMean3 = means(3, :) - stdevs(3, :);

topMean4 = means(4, :) + stdevs(4,:);
bottomMean4 = means(4, :) - stdevs(4, :);

xValues = linspace(-40, 60, 300);
disp(xValues);

%plot(topMean1); hold on;
%plot(bottomMean1);

fill([xValues, fliplr(xValues)], 20*[topMean1, bottomMean1], [0.75 0.75 1]); hold on;
plot(xValues, 20*means(1, :), 'color', 'b', 'LineWidth', 4); hold on;
%plot(xValues, topMean1, 'color', 'w'); hold on;
%plot(xValues, bottomMean1, 'color', 'w'); hold on;


fill([xValues, fliplr(xValues)], 20*[topMean2, bottomMean2], [0.875 1 0.875]); hold on;
plot(xValues, 20*means(2, :), 'color', 'g', 'LineWidth', 4); hold on;
%plot(xValues, topMean2, 'color', 'w'); hold on;
%plot(xValues, bottomMean2, 'color', 'w'); hold on;

fill([xValues, fliplr(xValues)], 20*[topMean3, bottomMean3], [0.875 1 1]); hold on;
plot(xValues, 20*means(3, :), 'color', 'c', 'LineWidth', 4); hold on;
%plot(xValues, topMean3, 'color', 'w'); hold on;
%plot(xValues, bottomMean3, 'color', 'w'); hold on;

fill([xValues, fliplr(xValues)], 20*[topMean4, bottomMean4], [1 0.875 1]); hold on;
plot(xValues, 20*means(4, :), 'color', 'm', 'LineWidth', 4); hold on;
%plot(xValues, topMean4, 'color', 'm', 'LineWidth', 1); hold on;
%plot(xValues, bottomMean4, 'color', 'm', 'LineWidth', 1); hold on;

xlabel('X Position');
ylabel('Z Position');

            %plot([0 1], [0 0], 'k', 'LineWidth', 7);
plot([-40 0], [1.5 1.5], 'k', 'LineWidth', 10);
plot([20 60], [1.5 1.5], 'k', 'LineWidth', 10);
text(9,4,'\downarrow', 'FontSize', 20)
text(1.5,12,'Baseline', 'FontSize', 20); 

set(gca, 'FontSize', 20);
set(gcf,'color','w');