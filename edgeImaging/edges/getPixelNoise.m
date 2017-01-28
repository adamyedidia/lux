function [px_std, avg_noise] = getPixelNoise(frames)
% get pixel noise over frames
% std of each pixel
px_std = std(frames, 0, ndims(frames));
avg_noise = mean(px_std);
end