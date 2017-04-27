
function grResize()
global figlist;

set(figlist.fig,'Units','pixels');
position = get(figlist.fig, 'Position');


x = position(1);
y = position(2);
w = position(3);
h = position(4);

set(figlist.list, 'Units', 'pixels', ...
                  'Position', [15, 15, w-30, h-56]);

set(figlist.uiText, 'Units', 'pixels', ...
                    'Position', [20, h-35, 160, 25]);

set(figlist.uiBtn, 'Units', 'pixels', ...
                   'Position',  [200, h-32, 100, 25]);

