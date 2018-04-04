function SurroundMat = CentreCircularZero(SurroundMat, CentreRadius)
%CentreCircularZero  sets the centre with given radius to 0.
%
% inputs
%   SurroundMat   the matrix that its centre will be set to 0.
%   CentreRadius  the radius of centre.
%
% outputs
%   SurroundMat  the centred set to 0 matrix.
%

[ws, hs] = size(SurroundMat);

if CentreRadius == 0 || CentreRadius > min(ws, hs)
  return;
end

x = max(ws, hs);
if mod(x, 2) == 0
  x = x + 1;
end
[rr, cc] = meshgrid(1:x);
centre = ceil(x / 2);
ch = sqrt((rr - centre) .^ 2 + (cc - centre) .^ 2) <= CentreRadius;

SurroundMat(ch) = 0;

end
