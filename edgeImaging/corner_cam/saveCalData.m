function outfile = saveCalData(overwrite, moviefile, gridfile, backfile, start)
% gets and saves the calibration data (rectification and corner) to outfile
addpath(genpath('../rectify'));

split = strsplit(moviefile, '.'); % the same name, with mat extension
outfile = strcat(split(1), '.mat');

if ~overwrite && exist(outfile, 'file')
    return
end

if nargin < 5
    start = 60*3; % starts 3 seconds in
end

v = VideoReader(moviefile);
endframe = v.NumberOfFrames;
frame1 = double(read(v,start));
if nargin < 4
    % use the mean of the moviefile for the background
    background = zeros(size(frame1));
    count = 0;
    for n = start:endframe
        background = background + double(read(v,n));
        count = count + 1;
    end
    background = background / count;
else
    vback = VideoReader(backfile);
    background = double(read(vback, floor(vback.NumberOfFrames/2)));
end
mean_pixel = mean(mean(background, 1), 2);

% solve for rectification
vcali = VideoReader(gridfile);
caliImg = read(vcali,100);
[iold, jold, ii, jj, ~] = rectify_image_solve(caliImg);
frame1 = rectify_image(frame1, iold, jold, ii, jj);
background = rectify_image(background, iold, jold, ii, jj);

figure; imagesc(frame1(:,:,1));
corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

clear('v', 'vback', 'vcali'); % clear video readers
clear('frame1', 'caliImg');
save(outfile);
end