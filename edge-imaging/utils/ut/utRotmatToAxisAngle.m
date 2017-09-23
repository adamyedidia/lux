
function [axis, ang] = utRotmatToAxisAngle(rotmat)

% utRotmatToAxisAngle:  3D rotation utility function
%
%   [axis, angle] = utRotmatToAxisAngle(rotmat)
%
% translates a pure-rotation matrix into an axis/angle representation.
%   "rotmat" should be a [3x3] matrix with the property that when it
%   is multiplied by a vector in this fashion:
%
%   [X', Y', Z'] = [X, Y, Z] * rotmat;
%
% it returns the result of taking [X, Y, Z] and rotating it about 
% some axis by some angle using the right-hand rule.
% 
% This function returns the "axis" (as a [1x3] row vector) and "angle" 
% of the rotation affected by such a matrix.
%
% This function is nominally the inverse of "utAxisAngleToRotmat",
% however because a rotation by "angle" around "axis" is the same as one
% by "-angle" around "-axis", or by "angle-2pi" around "axis", etc;
% it is not guaranteed that you'll get the same axis/angle pair
% back if you convert it to and from a rotation matrix with these functions.


ang = acos((trace(rotmat)-1)/2);

if (ang==0) axis = [1 0 0]; else
  twoSa = -2*sin(ang);
  x = (rotmat(3,2)-rotmat(2,3))/twoSa;
  y = (rotmat(1,3)-rotmat(3,1))/twoSa;
  z = (rotmat(2,1)-rotmat(1,2))/twoSa;
  axis = [x y z];  
end;

if (ang<0) ang=-ang; axis=-axis; end;
