
function im = utReadPnm(fname)

%  function im = utReadPnm(filename)
%
%  Opens the file and tries to read a pnm image out of it.
%  Returns HxWx3 or HxW arrays of values ranging 0..1.
%  One-Bit PNM files (.pbm) return logical arrays of zeros and ones.
%

[fid,msg] = fopen(fname,'r');
if (fid==-1) error(msg); end;

key = char(fread(fid,3));
switch (key(1:2)')
  case 'P6'; raw=1; depth=24;
  case 'P5'; raw=1; depth=8;
  case 'P4'; raw=1; depth=1;
  case 'P3'; raw=0; depth=24;
  case 'P2'; raw=0; depth=8;
  case 'P1'; raw=0; depth=1;
  otherwise; error('File doesn''t start with P[1-6] like PNMs should???');
end;

% -------------------------------------------------------------------

if (depth > 1) 

   dims = [];
   while (length(dims)<3)
      readDims = sscanf(getLine(fid),'%d');
      dims = [dims, reshape(readDims, 1, length(readDims))];
   end;

   maxval = dims(3);
   dims = dims(1:2);

   % images (P[2356])
   % ---------------------------------
   if (raw) data = fread(fid,prod(dims)*depth/8);
       else data = fscanf(fid,'%d',prod(dims)*depth/8);
   end;
   im = permute(reshape(data,[depth/8, dims(1), dims(2)]),[3 2 1]) / maxval;

else

   dims = [];
   while (length(dims)<2)
      readDims = sscanf(getLine(fid),'%d');
      dims = [dims, reshape(readDims, 1, length(readDims))];
   end;

   % bitmaps (P[14])
   % ---------------------------------
   
   if (~raw) data = fscanf(fid,'%s',prod(dims)) - '0';

        else data = fread(fid,floor((prod(dims)+7)/8));
             dex=zeros(size(data,1),8);  pwr=2.^(7:-1:0);             
             for col = 1:8; 
                dex(:,col) = floor(data ./ pwr(col));
                data = data - dex(:,col) * pwr(col);
             end;
             data = dex';                
   end;
   im = logical(reshape(data, dims(1),dims(2))');

end;

fclose(fid);


% -------------------------------------------------------
% read a line ignoring #-preceded comment lines.
% -------------------------------------------------------

function line = getLine(fid)

line='#';   
while (isstr(line) & (line(1)=='#'))
  line = fgetl(fid);
end;




