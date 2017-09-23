
function res = utIsSingleString (value)

res = 0;

if (ischar(value))
   if (length(size(value))==2)
      if (size(value,1)==1)
         res = 1;
      end;
   end;
end;
