
function rotmat = utAxisAngleToRotmat(axis,ang)

% utAxisAngleToRotmat:  3D rotation utility function
%
%   rotmat = utAxisAngleToRotmat(axis,angle)
%
% builds a rotation matrix given an axis and angle.  
%  "axis"  must be [1x3], a 3D row vector; it need not be normalized;
%  "angle" must be a scalar, an amount to rotate in radians.
%
% then "rotmat" will be the corresponding [3x3] rotation matrix 
% that may be used like this:
%
%   [X', Y', Z'] = [X, Y, Z] * rotmat;
%
% to rotate a vector [X, Y, Z] around "axis" by "angle" radians
% using the "right hand rule", in that if you were to point 
% your right-hand thumb in the direction of "axis", your fingers
% would curl in the direction of rotation.


axis = utNormalize(axis);
x=axis(1); y=axis(2); z=axis(3);
ca=cos(ang); sa=-sin(ang);
rotmat = [x*x*(1-ca)+ ca,   x*y*(1-ca)- sa*z, x*z*(1-ca)+ sa*y; ...
          y*x*(1-ca)+ sa*z, y*y*(1-ca)+ ca,   y*z*(1-ca)- sa*x; ...
          z*x*(1-ca)- sa*y, z*y*(1-ca)+ sa*x, z*z*(1-ca)+ ca];

