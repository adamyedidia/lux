function [corner_idx, wall_line_theta] = findWall(img, thetas, plot)
if nargin < 3
    plot = 0;
end
% the corner is always at (1,1), the wall is at a thetas(1) angle away
if abs(cos(thetas(1))) < 1e-4 % starting from k * pi/2 (k odd)
    wall_angle = pi/2;
else % starting from k * pi
    wall_angle = 0;
end
BW = edge(img(:,:,1), 'canny');
% figure; imshow(BW);
[H, T, R] = hough(BW, 'RhoResolution', 0.5, 'Theta', -90:0.5:89);
P = houghpeaks(H, 2, 'threshold', ceil(0.2*max(H(:))));

lines = houghlines(BW, T, R, P, 'FillGap', 10,'MinLength', 8);
wall_line_idx = 1;
wall_line_theta = 2*pi;
corner_idx = [1, 1];
corner_dist = size(img, 1) * sqrt(2);

if plot
    figure; imagesc(img(:,:,1)); hold on;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
        % Plot beginnings and ends of lines
        plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
        plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
    end
end

for k = 1:length(lines)
    theta = atan2(lines(k).point2(2) - lines(k).point1(2),...
        lines(k).point2(1) - lines(k).point1(1));
    dist1 = sqrt(lines(k).point1(1)^2 + lines(k).point1(2)^2);
    dist2 = sqrt(lines(k).point2(1)^2 + lines(k).point2(2)^2);
    if abs(theta - wall_angle) < abs(wall_line_theta - wall_angle)...
            && min(dist1, dist2) < corner_dist
        wall_line_idx = k;
        wall_line_theta = theta;
    end
end
wall_line_theta = thetas(1) + wall_line_theta - wall_angle;
wall_line_xmin = min(lines(wall_line_idx).point1(1), lines(wall_line_idx).point2(1));
wall_line_ymin = min(lines(wall_line_idx).point1(2), lines(wall_line_idx).point2(2));
if wall_line_xmin <= 3 % two pixels to the right
    corner_idx(1) = wall_line_xmin + 1;
end
if wall_line_ymin <= 3 % two pixels down
    corner_idx(2) = wall_line_ymin;
end % otherwise assume it's good at (1, 1)
end