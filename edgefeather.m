function[frout] = edgefeather(base,top,range)
    gf = ones(range)/range^2;
    bg = imfilter(base,gf,'replicate');
    topbw = imbinarize(rgb2gray(top),0);
    se1 = strel('disk',range*2);
    se2 = strel('disk',range);
    topbw_d = imdilate(topbw,se1);
    topbw_c = imerode(topbw_d,se2);
    topbw_e = imerode(topbw,se2);
    edge = topbw_c-topbw_e;
%     figure;
%     imshow(edge);
    frout(:,:,1) = bg(:,:,1).*edge + base(:,:,1).*(1-edge);
    frout(:,:,2) = bg(:,:,2).*edge + base(:,:,2).*(1-edge);
    frout(:,:,3) = bg(:,:,3).*edge + base(:,:,3).*(1-edge);
end