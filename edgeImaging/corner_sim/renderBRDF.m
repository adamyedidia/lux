function [imOut] = renderBRDF(clr, toLight, toCamera, surfaceNormal, brdf);
% renderBRDF renders a surface point as seen by the camera and as
% illuminated by the light in direction toLight.
% toLight may be a 2-d array of 3-element column vectors.
% toCamera is a 3-element column vector, as is surfaceNormal.

% Dec. 8, 2016  billf created.


switch brdf
    case 'Lambertian'
        if (ndims(toLight) == 2)
            tmp = toLight .* surfaceNormal;
            if tmp < 0
                imOut = 0 * clr;
            else
                imOut = tmp * clr;
            end
        else
            imOut = sum(toLight .* repmat(reshape(surfaceNormal,[1,1,3]),...
                [size(toLight,1),size(toLight,2),1]), 3);
            imOut = double((imOut > 0) .* imOut + (imOut < 0) * 0);
            imOut = double(repmat(reshape(clr,[1,1,3]), [size(imOut,1), size(imOut,2), 1])) .* ...
                repmat(imOut, [1,1,3]);
        end
end

