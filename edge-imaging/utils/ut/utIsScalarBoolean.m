
function res = utIsScalarBoolean (value)

res = 0;

if (isnumeric(value))
   if (max(size(value))==1)
      if ((value == 1) | (value == 0))
         res = 1;
      end;
   end;
end;
