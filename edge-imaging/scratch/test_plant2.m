addpath(genpath('matlabPyrTools'));

close all; clear;

names = {'MVI_2633', 'zoomedMVI_4958',...
    'reenactmentMVI_4959', 'pt_light_source_MVI_4960'};
resclips = [20, 5, 5, 60];
dtclips = [2, 2, 2, 10];
resxyclips = [0.5, 0.3, 0.2, 6];
dtxyclips = [0.1, 0.2, 0.1, 1];
% resxyclips = resclips/5;
% dtxyclips = dtclips/5;
vid = 1;

sigma = 1;
G1 = fspecial('gaussian',[4*sigma, 4*sigma], sigma);
[G1x,G1y] = gradient(G1);

sigma = 2;
G2 = fspecial('gaussian', [4*sigma, 4*sigma], sigma);
[G2x, G2y] = gradient(G2);

sfilt = binomialFilter(5);

width = 20;
fsize = width/2;
tfilt = zeros([1,1,1,fsize]);
f = binomialFilter(width);
tfilt(1,1,1,:) = f(1:fsize) * 2;

for n = 1:length(names)
    name = names{n};
    fprintf('processing %s\n', name);
    vidfile = sprintf('~/Downloads/%s.MOV', name);

    vsrc = VideoReader(vidfile);

    start_t = 5;
    end_t = vsrc.Duration - 1;
    fps = vsrc.FrameRate;
    ts = start_t: 1/fps: end_t;

    downlevs = 2;
    medimg = getMedianFrame(vidfile, downlevs, fps, start_t, end_t);
    [nrows, ncols, nchans] = size(medimg);
    frame_win = zeros([size(medimg), fsize]);

    if vid
        vout = VideoWriter(sprintf('spatial_%s.avi', name));
        vout.FrameRate = 10;
        open(vout);
    else
        ts = ts(1:fsize+20);
    end
    
    prevframe = zeros(size(medimg));
    for i = 1:length(ts)
        vsrc.CurrentTime = ts(i);
        curframe = blurDnClr(double(readFrame(vsrc)), downlevs, sfilt);
        frame_win(:,:,:,1:end-1) = frame_win(:,:,:,2:end);
        frame_win(:,:,:,end) = curframe;
        if i < fsize
            continue;
        end
        frame = sum(bsxfun(@times, tfilt, frame_win), 4);
        resx = zeros(size(frame));
        resy = zeros(size(frame));
        resxy = zeros(size(frame));
        dtx = zeros(size(frame));
        dty = zeros(size(frame));
        if i > fsize + 1
            res = frame - medimg;
            mclip = resclips(n);
            resclip = (res + mclip)/2/mclip;

            dt = frame - prevframe;     
            mclip = dtclips(n);
            dtclip = (dt + mclip)/2/mclip;
            im0 = vertcat(resclip, imgaussfilt(dtclip, 2));

            for c = 1:size(frame,3)
                resx(:,:,c) = corrDn(res(:,:,c), G2x, 'reflect1');
                resy(:,:,c) = corrDn(res(:,:,c), G2y, 'reflect1');
                resxy(:,:,c) = corrDn(resy(:,:,c), G2x, 'reflect1');
                dtx(:,:,c) = corrDn(dt(:,:,c), G1x, 'reflect1');
                dty(:,:,c) = corrDn(dt(:,:,c), G1y, 'reflect1');
            end
            mclip = resxyclips(n);
            resxclip = (resx + mclip)/mclip/2;
            resyclip = (resy + mclip)/mclip/2;
            
            mclip = resxyclips(n)/10;
            resxyclip = (resxy + mclip)/mclip/2;
            imres = horzcat(resxclip, resyclip);

            mclip = dtxyclips(n);
            dtxclip = (dtx + mclip)/mclip/2;
            dtyclip = (dty + mclip)/mclip/2;
                        
            imdt = horzcat(dtxclip, dtyclip);
            im1 = vertcat(imres, imdt);

            im = horzcat(im0, im1);            
            imorig = vertcat(curframe/255, frame/255);
            
%             outim = horzcat(imorig, im);
            
            im2 = vertcat(resclip, resyclip);
            im3 = vertcat(resxclip, resxyclip);
            outim = horzcat(im2, im3);
            if vid
                writeVideo(vout, uint8(255*outim));
            else
                imshow(outim); pause(0.1);
            end
        end

        prevframe = frame;
    end
    if vid
        close(vout);
    end

end