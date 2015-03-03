clear

fx = imread('res/flowx.png');
fy = imread('res/flowy.png');


from = imread('../input/res256_01.bmp');
to = imread('../input/res256_02.bmp');
%}
%{
from = imread('../input/test1.png');
to = imread('../input/test2.png');
%}

fx = double(fx);
fy = double(fy);

imshow(to)
hold
quiver(fx, fy, 0)

figure
imshow(from)
hold
quiver(fx, fy, 0)

figure
imshow(to - from)
hold
quiver(fx, fy, 0)