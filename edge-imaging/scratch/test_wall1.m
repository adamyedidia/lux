% clear; close all;

mydir = fileparts(mfilename('fullpath'));
rootdir = fileparts(mydir);

addpath(genpath(fullfile(rootdir, 'rectify')));
addpath(genpath(fullfile(mydir, 'matlabPyrTools')));

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/wallImaging/2017-09-13/';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');

filter = binomialFilter(5);

back = blurDnClr(double(imread(fullfile(expfolder, 'back1.ppm'))), 3, filter);
homfile = fullfile(expfolder, 'wall_rectify.mat');
if 0%exist(homfile, 'file')
    load(homfile, 'iold', 'jold', 'ii', 'jj');
else
    refdims = [1,1];
    refim = blurDnClr(double(imread(fullfile(expfolder,...
        'calibration.ppm'))), 3, filter);
    [iold, jold, ii, jj, ~] = rectify_image_solve(refim, refdims);
    save(homfile, 'iold', 'jold', 'ii', 'jj');
end

rect_back = rectify_image(back, iold, jold, ii, jj);

loc_a = blurDnClr(double(imread(fullfile(expfolder, 'loc_a1.ppm'))), 3, filter);
rect_a = rectify_image(loc_a, iold, jold, ii, jj);
figure; imshow((rect_back - rect_a)/3e2);

loc_b = blurDnClr(double(imread(fullfile(expfolder, 'loc_b1.ppm'))), 3, filter);
rect_b = rectify_image(loc_b, iold, jold, ii, jj);
figure; imshow((rect_back - rect_b)/3e2);

loc_c = blurDnClr(double(imread(fullfile(expfolder, 'loc_c1.ppm'))), 3, filter);
rect_c = rectify_image(loc_c, iold, jold, ii, jj);
figure; imshow((rect_back - rect_c)/3e2);

loc_d = blurDnClr(double(imread(fullfile(expfolder, 'loc_d1.ppm'))), 3, filter);
rect_d = rectify_image(loc_d, iold, jold, ii, jj);
figure; imshow((rect_back - rect_d)/3e2);

loc_e = blurDnClr(double(imread(fullfile(expfolder, 'loc_e1.ppm'))), 3, filter);
figure; imshow((back - loc_e)/3e2);
