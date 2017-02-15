function outfile = saveCalData(overwrite, moviefile, gridfile, backfile, start)
% gets and saves the calibration data (rectification and corner) to outfile
addpath(genpath('../rectify'));

split = strsplit(moviefile, '.'); % the same name, with mat extension
outfile = char(strcat(split(1), '.mat'));

if ~overwrite && exist(outfile, 'file')
    return
end

if nargin < 4
    start = 60*5; % starts 5 seconds in
end

v = VideoReader(moviefile);
endframe = v.NumberOfFrames;
frame1 = double(read(v,start));
framesize = size(frame1);

avg_img = zeros(size(frame1));
count = 0;
for n = start:5:endframe
    avg_img = avg_img + double(read(v,n));
    count = count + 1;
end
avg_img = avg_img / count;
mean_pixel = mean(mean(avg_img, 1), 2);

vback = VideoReader(backfile);
background = double(read(vback, floor(vback.NumberOfFrames/2)));

% solve for rectification
vcali = VideoReader(gridfile);
caliImg = read(vcali,100);
[iold, jold, ii, jj, ~] = rectify_image_solve(caliImg);
frame1 = rectify_image(frame1, iold, jold, ii, jj);
background = rectify_image(background, iold, jold, ii, jj);

clear('v', 'vback', 'vcali', 'caliImg'); % clear unnecessary variables
save(outfile);
end