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
load(params.mean_datafile, 'avg_img');

avg_img = rectify_image(avg_img, iold, jold, ii, jj);
avg_img = blurDnClr(avg_img, params.downlevs, params.filt);
figure; imagesc(avg_img(:,:,1));

corners = ginput(ncorners);
hold on; plot(corners(:,1), corners(:,2), 'ro');

% save corners and rectified, downsampled average image
save(outfile, 'corners', 'avg_img');
end