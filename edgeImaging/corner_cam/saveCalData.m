function outfile = saveCalData(overwrite, gridfile)
% gets and saves the calibration data (rectification) to outfile

addpath(genpath('../rectify'));

split = strsplit(gridfile, '.'); % the same name, with mat extension
outfile = char(strcat(split(1), '.mat'));

if ~overwrite && exist(outfile, 'file')
    return
end

% solve for rectification
vcali = VideoReader(gridfile);
caliImg = read(vcali,100);
[iold, jold, ii, jj, ~] = rectify_image_solve(caliImg);

save(outfile, 'iold', 'jold', 'ii', 'jj');
end