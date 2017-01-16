
function image = utCropImageToPowerOfTwo (image, powers)

totalPaddingRequired = power(2,powers);

for dim = [1 2]
  rem = mod(size(image,dim),totalPaddingRequired);
  ptS(dim)=1;
  ptE(dim)=size(image,dim);

  % This computes by how much to crop the image to ensure that its
  % size is a multiple of the required number of powers of two.

  if (rem ~= 0)
     if (size(image,dim) <= rem) 
        error('image too small');
     end;
     remL = floor(rem/2);
     remR = rem - remL;
     ptS(dim) = 1+remL;
     ptE(dim) = size(image,dim)-remR;
  end;
end;

image = image(ptS(1):ptE(1),ptS(2):ptE(2),:);
