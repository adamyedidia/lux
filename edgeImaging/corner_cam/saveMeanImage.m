function outfile = saveMeanImage(overwrite, moviefile, start, step)
% compute the mean image of the moviefile, starting from startframe
% optional step size, for convenience/speed reasons

split = strsplit(moviefile, '.'); % the same name, with mat extension
outfile = char(strcat(split(1), '_mean_img.mat'));

if ~overwrite && exist(outfile, 'file')
    return
end

if nargin < 4
    step = 30;
end

if nargin < 3
    start = 60*5; % starts 5 seconds in
end

v = VideoReader(moviefile);
endframe = v.NumberOfFrames - start

frame1 = double(v.read(start));
avg_img = zeros(size(frame1));

count = 0;
for n = start:step:endframe
    fprintf('Frame %i\n', n);
    avg_img = avg_img + double(v.read(n));
    count = count + 1;
end
avg_img = avg_img / count;
mean_pixel = mean(mean(avg_img, 1), 2);

save(outfile, 'avg_img', 'mean_pixel', 'endframe');
end
