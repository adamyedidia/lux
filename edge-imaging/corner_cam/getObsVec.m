function [y, nanrows] = getObsVec(frame, params)
if strcmp(params.amat_method, 'interp')
    rgbq = gradientAlongCircle(frame, params.corner, ...
        params.rs, params.nsamples, params.theta_lim);
    
    rgbq = permute(rgbq, [1, 3, 2]); % rgbq is nsamples x nchans x length(rs)
    y = reshape(rgbq, [size(rgbq,1)*size(rgbq,2), size(rgbq,3)]);
    nanrows = any(isnan(y),2);
    y(nanrows,:) = []; % delete nanrows
else % default is all pixels
    nchans = size(frame, 3);    
    y = zeros([params.outr^2, nchans]);
    for c = 1:nchans
        chan = frame(:,:,c);
        y(:,c) = reshape(chan(params.crop_idx), [params.outr^2, 1]);
    end
    nanrows = any(isnan(y),2);
    y(nanrows,:) = []; % delete nanrows
end
end
