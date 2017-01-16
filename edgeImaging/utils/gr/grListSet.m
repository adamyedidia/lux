
function grListSet(element,str)
global figlist;

if (~figlist.deleted)

   n=length(str); width=figlist.nChars;
   if (n > width) str=str(1:width);
             else str(n+1:width) = double(' ');
   end;
   
   currentList = get(figlist.list, 'String');
   currentList(element,:) = str;
   set(figlist.list, 'String', currentList);

end;