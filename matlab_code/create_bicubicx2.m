clear

test_set = 'shake';

a = imread(['../input/' test_set 'Crop2000_01.bmp']);
b = imresize(a, 2, 'bicubic');
c = imresize(a, 4, 'bicubic');

imwrite(b, ['../output/compare/' test_set 'Crop2000_bicubic_x2.bmp']);
imwrite(c, ['../output/compare/' test_set 'Crop2000_bicubic_x4.bmp']);