 a = imread('res2000_LinearConstruct_HR4_Gaussian0.bmp');
 b = a(1539:1711, 1439:1575);
 %b = a(2107:2299, 1779:2287);
 %b = a(1311:1451, 2203:2311);
 
 imwrite(b, 'zz0.bmp');