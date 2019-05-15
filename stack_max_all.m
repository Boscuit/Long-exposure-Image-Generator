%---------stack method max filter(pixel by pixel) stack all---------
function[frout] = stack_max_all(frgraylist,frlist)
    frout = frlist(:,:,:,1);
    for i = 1:size(frout,1)
        for j = 1:size(frout,2)
            label = find(frgraylist(i,j,:)==max(frgraylist(i,j,:)));
            num = label(1);%the first frame have max graylevel 
            frout(i,j,:) = frlist(i,j,:,num);%assignment
        end
    end
end