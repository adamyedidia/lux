function [outframes, uangles] = avgRepeatColumns(inframes, angles)
% go through angles and average columns
[uangles, ~, i1] = unique(angles, 'stable');
outframes = zeros([size(inframes,1), length(uangles), size(inframes,3)]);
for i = 1:length(uangles)
    % i==i1 is idx of angles equal to langle(i)
    outframes(:,i,:) = mean(inframes(:,i==i1,:), 2);
end
end