
function grSetVisible (figs,visible)
global figlist;

if (visible) tag='#'; code='on'; 
        else tag=' '; code='off'; end;

currentList = get(figlist.list, 'String');
for fig = figs
   if (currentList(fig-1,7) == double(':'))
      currentList(fig-1,6) = tag;
      set(fig,'Visible',code);
   end;
end;
set(figlist.list, 'String',  currentList);

