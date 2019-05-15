%---------stack method median filter(pixel by pixel)---------
%to function properly, the frame number should be odd 
function[frout] = stack_median_all(frgraylist,frlist)
    frout = frlist(:,:,:,1);
    for i = 1:size(frout,1)
        for j = 1:size(frout,2)
            label = find(frgraylist(i,j,:)==median(frgraylist(i,j,:)));
            num = label(1);%the first frame have min graylevel 
            frout(i,j,:) = frlist(i,j,:,num);%assignment
        end
    end
end