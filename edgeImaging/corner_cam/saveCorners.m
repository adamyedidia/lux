function outfile = saveCorners(overwrite, moviefile, ncorners, params)
% gets the corners for scene in moviefile
% saves a different set of corners for different nlevs of downsampling
addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));

split = strsplit(moviefile, '.'); % the same name, with suffix
suffix = sprintf('_ncorners=%d_downlevs=%d.mat', ncorners, params.downlevs);
outfile = char(strcat(split(1), suffix));

if ~overwrite && exist(outfile, 'file')
    return
end

load(params.cal_datafile); % for iold, jold, ii, jj
v = VideoReader(moviefile);
frame1 = double(read(v,params.start));

frame1 = rectify_image(frame1, iold, jold, ii, jj);
frame1 = blurDnClr(frame1, params.downlevs, params.filt);
figure; imagesc(frame1(:,:,1));

corners = ginput(ncorners);
hold on; plot(corners(:,1), corners(:,2), 'ro');

% save corners and rectified, downsampled average image
save(outfile, 'corners', 'avg_img');
end