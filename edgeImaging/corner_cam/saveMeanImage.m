function outfile = saveMeanImage(overwrite, moviefile, startframe, step)
% compute the mean image of the moviefile, starting from startframe
% optional step size, for convenience/speed reasons

split = strsplit(moviefile, '.'); % the same name, with mat extension
outfile = char(strcat(split(1), '_mean_img.mat'));

if ~overwrite && exist(outfile, 'file')
    return
end

v = VideoReader(moviefile);
endframe = v.NumberOfFrames;

if nargin < 4
    step = 10;
end

if nargin < 3
    startframe = 60*5; % starts 5 seconds in
end

frame1 = double(read(v,startframe));
avg_img = zeros(size(frame1));

count = 0;
for n = startframe:step:endframe
    avg_img = avg_img + double(read(v,n));
    count = count + 1;
end
avg_img = avg_img / count;
mean_pixel = mean(mean(avg_img, 1), 2);

save(outfile, 'avg_img', 'mean_pixel', 'endframe');
end