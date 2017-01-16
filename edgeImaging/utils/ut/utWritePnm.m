
function utWritePnm(im,fname,varargin)

%  function utWritePnm(im,fname,varargin)
%
%  Saves HxWx3 or HxW arrays of values ranging 0..1.
%  Saves as a one-bit PNM file if im is logical.
%
%  Options may include .. 
%     'raw',    1|0  -- write a binary/ascii ppm
%     'maxval', 255  -- ppm maxvals are always 255 
%
%
% Dec 31, 1998  ecp wrote
%

[raw, maxval] = utParseArgs(varargin,{
  { 'raw',    1   },
  { 'maxval', 255 }
});


if ((min(min(min(im)))<0) | (max(max(max(im)))>1))
  % Well the spec for this function says that images must
  % be in the range 0-1.  This image isn't.  But rather than
  % being all hard-line and returning with an error message,
  % we'll try our best to save something reasonable.  
  
  mn = min(min(min(im)));
  mx = max(max(max(im)));

  if ((mn>=0) & (mx<=256) & (mx-mn > 40))
    % Perhaps this is a regular image with a 0-255 scale; and
    % the caller just forgot to rescale it like he should've?

    im = im/256;

  elseif (mx-mn < eps) 
    % Whatever..

    im(:,:,:) = 0;

  else
    % Well we haven't a clue what the ideal range is.  So we'll
    % use the image's min and max..

    im = (im-mn)/(mx-mn);
  end;
end;


if (islogical(im)) depth = 1; 
              else depth = 8*size(im,3);  
end;

switch (depth)
  case 24;   code='P3';
  case  8;   code='P2';
  case  1;   code='P1';
  otherwise; error('image not depth 1 or 3???');
end;
if (raw) code(2)=code(2)+3; end;

[fid,msg] = fopen(fname,'w');
if (fid==-1) error(msg); end;

fprintf(fid,'%s\n# Written by Matlab''s "utWritePnm.m"\n%d %d\n', ...
        code, size(im,2), size(im,1));

% -------------------------------------------------------------------

if (depth>1)

   % images (P[2356])
   % ---------------------------------

   data = floor(permute(im,[3 2 1]) * (maxval+.999));
   fprintf(fid,'%d\n',maxval);
   if (raw) fwrite(fid,data);
       else fprintf(fid,'%d %d %d\n',data);
   end;

else

   % bitmaps (P[14])
   % ---------------------------------

   if (~raw)

     eoln = sprintf('\n');
     data = [ char(im'+'0'); (ones(1,size(im,1))*eoln(1)) ];

   else

     bytes = floor((prod(size(im))+7)/8);
     data  = zeros(8*bytes,1);
     data(1:prod(size(im))) = reshape(im',1,prod(size(im)));
     bgrid = reshape(data,8,bytes)';
     
     data = zeros(bytes,1); pwr=2.^(7:-1:0); 
     for col = 1:8; data=data + bgrid(:,col)*pwr(col); end;

   end;
   fwrite(fid,data);

end;

fclose(fid);
