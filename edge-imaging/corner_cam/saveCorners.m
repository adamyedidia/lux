function outfile = saveCorners(overwrite, isvid, moviefile, ncorners, params)
% gets the corners for scene in moviefile
% saves a different set of corners for different nlevs of downsampling
addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));

suffix = sprintf('ncorners=%d_downlevs=%d.mat', ncorners, params.downlevs);

switch isvid
case 1 % is a video
    split = strsplit(moviefile, '.'); % the same name, with suffix
    outfile = char(strcat(split(1), '_', suffix));
    vsrc = VideoReader(moviefile);
    avg_img = read(vsrc, params.start);
case 0 % is a directory
    outfile = fullfile(moviefile, suffix);
    avg_img = imread(fullfile(moviefile, 'photo_1_2.CR2'));
case -1 % pointgrey directory
    outfile = sprintf('%s_%s', moviefile, suffix);
    fname = sprintf('%s-%4.4d.pgm', moviefile, 1);
    avg_img = demosaic(imread(fname), 'rggb');
end
    
if ~overwrite && exist(outfile, 'file')
    return
end

fprintf('Saving corners data in %s\n', outfile);

load(params.cal_datafile); % for iold, jold, ii, jj
try
    % if successful, overwrite the first frame
    load(params.mean_datafile, 'avg_img');
catch
    warning('no mean data file, using first frame of the video');
end

avg_img = rectify_image(avg_img, iold, jold, ii, jj);
avg_img = blurDnClr(avg_img, params.downlevs, params.filt);
figure; imagesc(avg_img(:,:,1));

corners = ginput(ncorners);
hold on; plot(corners(:,1), corners(:,2), 'ro');

% save corners and rectified, downsampled average image
save(outfile, 'corners', 'avg_img');
end
