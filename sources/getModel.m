function [ index ] = getModel(Ep, mi)
    
    metric(1:length(Ep)) = NaN;
    for i = 1:1:length(Ep)
        if isnan(Ep(i)) == false && isnan(mi(i)) == false 
            metric(i) = sqrt((Ep(i))^2 + (1 - mi(i))^2);
        else
            metric(i) = NaN;
        end
    end
    
    [~, index] = min(metric);
end

