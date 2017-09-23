
function res = utIsScalarInteger (value)

res = 0;

if (isnumeric(value))
   if (max(size(value))==1)
      res = 1;
   end;
end;
