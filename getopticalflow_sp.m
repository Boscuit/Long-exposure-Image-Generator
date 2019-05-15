%---------get optical flow with superpixel segementation---------
function [opticalflow] = getopticalflow_sp(im1,im2,threshold,parts)
    opticalflow = zeros([size(im1),2]);
    ww = 20;% window size ww*ww
    w = round(ww/2);
    %get superpixel and pad in boudrary with 0
    [L,N] = superpixels(im1,parts);% L superpixel-map with label 
    Lpad = zeros(size(L));
    Lpad(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1) = L(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1);%pad in boudary with 0
%     figure
%     BW = boundarymask(L);
%     imshow(imoverlay(im1,BW,'cyan'),'InitialMagnification',67)
    
    %get those pixels which have the same label
    IDXpad = label2idx(Lpad);%for calculation
    IDX = label2idx(L);%for assignment
    IDXfinalpad = cell(1,N);% to avoid last label being padded and cell lost
    for label = 1:size(IDXpad,2)
        IDXfinalpad{label} = IDXpad{label};
    end

    % Lucas Kanade Here
    % for each point, calculate I_x, I_y, I_t
    Ix_m = conv2(im2, [-1 1; -1 1], 'valid'); % partial on x
    Iy_m = conv2(im2, [-1 -1; 1 1], 'valid'); % partial on y
    It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t??
    u = zeros(size(im1));
    v = zeros(size(im1));
    
    % within window ww * ww
    for labelVal = 1:N
        allidxpad = IDXfinalpad{labelVal};
        allidx = IDX{labelVal};
        if size(allidxpad,1)~=0
            idx = allidxpad(round(size(allidxpad,1)/2)); %--middle one
            j = ceil(idx/size(L,1));
            i = idx - (j-1)*size(L,1);
            Ix = Ix_m(i-w:i+w, j-w:j+w);
            Iy = Iy_m(i-w:i+w, j-w:j+w);
            It = It_m(i-w:i+w, j-w:j+w);

            Ix = Ix(:);
            Iy = Iy(:);
            b = -It(:); % get b here

            A = [Ix Iy]; % get A here
            nu = pinv(A)*b; % get velocity here
            
            if norm(nu,2) > threshold % avoid extreme optical flow (estimate wrongly)
                nu = [0;0];
            end
            u(allidx)=nu(1);
            v(allidx)=nu(2);
        end
    end
    opticalflow(:,:,1) = u;
    opticalflow(:,:,2) = v;
end