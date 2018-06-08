function [ regularized_data, x_max, x_min, x_mean ] = regularize_data(data)
    x_max = max(data);
    x_min = min(data);
    x_mean = mean(data);
    regularized_data = (data - x_mean)./(x_max - x_min);
end

