
figure,
 E = poz;
 imshow(E, 'InitialMag', 'fit')
 
 I = -1*negative;
 imshow(I, 'InitialMag', 'fit')
 
 
 imshow(E, 'InitialMag', 'fit')
 % Make a truecolor all-green image.
 green = cat(3, I,... 
        zeros(size(E)),  zeros(size(E)));
 hold on 
 h = imshow(green); 
 hold off
 
 % Use our influence map as the 
 % AlphaData for the solid green image.
 set(h, 'AlphaData', I)