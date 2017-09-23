function outfile = saveCalData(overwrite, input_type, gridfile, refdims)
% gets and saves the calibration data (rectification) to outfile
% the gridfile is either a video or an image

if nargin < 4
    refdims = [1, 1];
end

split = strsplit(gridfile, '.'); % the same name, with mat extension
outfile = char(strcat(split(1), '.mat'));

if ~overwrite && exist(outfile, 'file')
    return
end

fprintf('Saving rectification data in %s\n', outfile);

% solve for rectification
switch input_type
case 0
    % single rectification image
    caliImg = imread(gridfile);
case 1
    vcali = VideoReader(gridfile);
    caliImg = read(vcali,100);
case -1
    fname = sprintf('%s-%4.4d.pgm', gridfile, 10);
    caliImg = demosaic(imread(fname), 'rggb');
end

[iold, jold, ii, jj, ~] = rectify_image_solve(caliImg, refdims);

save(outfile, 'iold', 'jold', 'ii', 'jj');
end
