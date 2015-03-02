clear

a = zeros(100,100);
b = zeros(100,100);

a(30:40,30:40) = 1;
b(30:40,40:50) = 1;

a(60:65,60:65) = 1;
b(60:65,60:65) = 1;

imwrite(a, '../input/test1.png');
imwrite(b, '../input/test2.png');