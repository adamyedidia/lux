function derivative_RTheta(srcfile, params)
params = setObsXYLocs(params);
vsrc = VideoReader(srcfile);

load(params.mean_datafile, 'back_img');
back_img = preprocessFrame(back_img, params);
mean_pixel = mean(mean(back_img, 1), 2);

frameidx = params.startframe: params.step: params.endframe;
times = frameidx / vsrc.FrameRate;

nchans = size(back_img, 3);
samples = zeros([size(params.obs_xlocs), nchans]);
[yy, xx] = ndgrid(1:size(back_img,1), 1:size(back_img,2));

figure;
for i = 1:length(times)
    fprintf('Frame %i\n', frameidx(i));
    vsrc.CurrentTime = times(i);
    framen = preprocessFrame(double(readFrame(vsrc)), params);
    res = bsxfun(@plus, framen - back_img, mean_pixel);
    for c = 1:nchans
        samples(:,:,c) = interp2(xx, yy, res(:,:,c),...
            params.obs_xlocs, params.obs_ylocs);
    end
    grad = diff(diff(samples, 2), 1);
    imagesc(grad/4); pause(0.1);
end
end