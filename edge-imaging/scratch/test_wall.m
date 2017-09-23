clear; close all;

mydir = fileparts(mfilename('fullpath'));
rootdir = fileparts(mydir);

addpath(genpath(fullfile(rootdir, 'rectify')));
addpath(genpath(fullfile(mydir, 'matlabPyrTools')));

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/wallImaging/2017-08-31/';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');

scenedir = fullfile(expfolder, 'monitor_lines');
backfile = fullfile(scenedir, 'back_dark.ppm');
backfile = fullfile(scenedir, 'back.ppm');
back = blurDnClr(double(imread(backfile)), 2, binomialFilter(5));

homfile = fullfile(mydir, 'wall_homo_coords.mat');
if exist(homfile, 'file')
    load(homfile, 'iold', 'jold', 'ii', 'jj');
else
    refdims = [3, 2];
    [iold, jold, ii, jj, ~] = rectify_image_solve(back, refdims);
    save(homfile, 'iold', 'jold', 'ii', 'jj');
end

backrect = rectify_image(back, iold, jold, ii, jj);
backrect = backrect(100:end/2, 200:1200, :);

loca = blurDnClr(double(imread(fullfile(scenedir,...
    'loc_a_dark.ppm'))), 2, binomialFilter(5));
loca = blurDnClr(double(imread(fullfile(scenedir,...
    'loc_a.ppm'))), 2, binomialFilter(5));
arect = rectify_image(loca, iold, jold, ii, jj);
arect = arect(100:end/2, 200:1200, :);

locb = blurDnClr(double(imread(fullfile(scenedir,...
    'loc_b_dark.ppm'))), 2, binomialFilter(5));
locb = blurDnClr(double(imread(fullfile(scenedir,...
    'loc_b.ppm'))), 2, binomialFilter(5));
brect = rectify_image(locd, iold, jold, ii, jj);
brect = brect(100:end/2, 200:1200, :);

figure; subplot(311); imagesc(backrect/5000); title('background'); axis off;
subplot(312); imagesc((backrect - arect)/500); title('loc A'); axis off;
subplot(313); imagesc((backrect - brect)/500); title('loc B'); axis off;

