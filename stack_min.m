%---------stack method min filter(pixel by pixel)---------
function[frout,frout_only] = stack_min(base,frgraylist,frlist,isfg)
    frNum = zeros(size(isfg,1),size(isfg,2));%selection record
    frout = base;
    frout_only = zeros(size(base));
    for i = 1:size(isfg,1)
        for j = 1:size(isfg,2)
            if isfg(i,j)~=0
                label = find(frgraylist(i,j,:)==min(frgraylist(i,j,:)));
                num = label(1);%the first frame have median graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
                frout_only(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end