datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb20';
resfolder = fullfile(datafolder, 'results');
track1_path = fullfile(resfolder, 'corner1_redblue_tracks.png');
track2_path = fullfile(resfolder, 'corner2_redblue_tracks.png');

track1 = fliplr(rgb2gray(imread(track1_path)));
track2 = rgb2gray(imread(track2_path));

% clean the tracks
track1 = medfilt2(track1);
track2 = medfilt2(track2);
figure; subplot(121); imagesc(track1); subplot(122); imagesc(track2);
