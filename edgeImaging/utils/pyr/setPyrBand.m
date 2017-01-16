% [outPyr, outPind] = setPyrBand(pyr, pind, level, band, coeffs)
%
% Set a subband from a pyramid (It may work for gaussian, laplacian, QMF/wavelet, 
% or steerable.  But only steerable tested.).    Note that band 1 of the pyramid seems to be the 
% image as reconstructed without the low-frequencies???

% Bill Freeman 11/10/00

function [outPyr, outPind] = setPyrBand(pyr, pind, level, band, coeffs)


nbands = spyrNumBands(pind);
if ((band > nbands) | (band < 1))
  error(sprintf('Bad band number (%d) should be in range [1,%d].', band, nbands));
end	

maxLev = spyrHt(pind);
if ((level > maxLev) | (level < 1))
  error(sprintf('Bad level number (%d), should be in range [1,%d].', level, maxLev));
end

firstband = 1 + band + nbands*(level-1);
if ((pind(firstband,1) ~= size(coeffs,1)) | ...
    (pind(firstband,2) ~= size(coeffs,2)))
  error('setPyrBand coeff size incorrect for desired band');
end

outPyr = pyr;

outPyr(pyrBandIndices(pind,firstband)) = reshape(coeffs,size(coeffs,1)* ...
					    size(coeffs,2),1);
outPind = pind;

