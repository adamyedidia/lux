% 2017-09-19 adamyedidia
clear; close all;

mydir = fileparts(mfilename('fullpath'));
rootdir = fileparts(mydir);

addpath(genpath(fullfile(rootdir, 'rectify')));
addpath(genpath(fullfile(mydir, 'matlabPyrTools')));

datafolder = '/Users/adamyedidia/Dropbox (MIT)/shadowImaging/wallImaging/2017-09-19/';
expfolder = fullfile(datafolder, 'videos');
resfolder = fullfile(datafolder, 'results');

vidtype = '.MP4';
calname = 'hall_calibration';
backname = 'hall_background';
vidname = 'hall_walking2';

start_t = 1;
end_t = 31;
process_fps = 10;

downlevs = 1;
filter = binomialFilter(5);

% read in frame from calibration file
% TODO: in future, we can use the mean/median frame of this video
vcal = VideoReader(fullfile(expfolder, strcat(calname, vidtype)));
vcal.CurrentTime = start_t; % avoid camera moving
calimg = blurDnClr(double(readFrame(vcal)), downlevs, filter);

cal_coords = fullfile(expfolder, strcat(calname, '.mat'));
if exist(cal_coords, 'file')
    load(cal_coords, 'iold', 'jold', 'ii', 'jj');
else
    refdims = [1, 1];
    [iold, jold, ii, jj, ~] = rectify_image_solve(calimg, refdims);
    save(cal_coords, 'iold', 'jold', 'ii', 'jj');
end

% read in background frame
% in future could use mean/median frame of this video, or the input video
vback = VideoReader(fullfile(expfolder, strcat(backname, vidtype)));
vback.CurrentTime = start_t; % avoid camera moving
backim = blurDnClr(double(readFrame(vback)), downlevs, filter);
% backrect = rectify_image(backim, iold, jold, ii, jj);

% now read in the video and optionally write out residual video file
vsrc = VideoReader(fullfile(expfolder, strcat(vidname, vidtype)));
vout = VideoWriter(fullfile(resfolder, strcat(vidname, '.avi')));
vout.FrameRate = process_fps;
open(vout);

for t = start_t: 1/process_fps: end_t
    vsrc.CurrentTime = t;
    frame = blurDnClr(double(readFrame(vsrc)), downlevs, filter);
%     framerect = rectify_image(frame, iold, jold, ii, jj);
%     imshow((backrect - framerect)/6); pause(0.1);
    writeVideo(vout, uint8((backim - frame)*30));
end

close(vout);