function [ deregularized_data ] = deregularize_data(data, x_max,  x_min, x_mean)
    deregularized_data = data.*(x_max - x_min) + x_mean;
end
