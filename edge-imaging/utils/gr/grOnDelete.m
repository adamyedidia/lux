
function grOnDelete()
global figlist

% Todo:  The gr FigureList window needs to support deleted
%        figures better than this!

if (~figlist.deleted)
  grListSet(gcbo-1,sprintf(' %02d   - - <<deleted>> - -',gcbo));
end;
