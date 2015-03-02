clear

fx = imread('bear/flowx.png');
fy = imread('bear/flowy.png');

from = imread('../input/bear256_01.bmp');
to = imread('../input/bear256_02.bmp');

from = double(from)/255;
to = double(to)/255;

warp = zeros(size(from));

for i = 1:size(from,1)
    for j = 1:size(from,2)
        warp(i+fy(i,j), j+fx(i,j),:) = from(i,j,:);
    end
end

imshow(warp)
figure
imshow(to)
figure
imshow(from)
figure
imshow(to - warp)