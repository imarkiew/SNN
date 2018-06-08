clear all;
close all;
clc;

% przyj�te maksymalnej liczby neuron�w ukrytych oraz liczby powt�rze�
max_number_of_hidden_nurons = 10;
number_of_iterations = 10;

% wczytywanie danych ucz�cych i testowych
train_data = load('zestaw_apr_2_train.txt');
test_data = load('zestaw_apr_2_test.txt');

% normalizacja danych ucz�cych i testowych
x_train_original = train_data(:, 1);
[x_train, ~, ~, ~] = regularize_data(x_train_original);
y_train_original = train_data(:, 2);
[y_train, y_train_max, y_train_min, y_train_mean] = regularize_data(y_train_original);
x_test_original = test_data(:, 1);
[x_test, ~, ~, ~] = regularize_data(x_test_original);
y_test_original = test_data(:, 2);

% budowanie modeli dla r�nej liczby neuron�w ukrytych oraz liczby
% powt�rze�
for i = 1:1:max_number_of_hidden_nurons
    for j = 1:1:number_of_iterations
        
        net = train_net(x_train, y_train, i);

        net_y_values_train(:, i, j) = deregularize_data(sim(net, x_train'), y_train_max, y_train_min, y_train_mean)';
        net_mse_train(:, i, j) = immse(net_y_values_train(:, i, j), y_train_original);

        net_y_values_test(:, i, j) = deregularize_data(sim(net, x_test'), y_train_max, y_train_min, y_train_mean)';
        net_mse_test(:, i, j) = immse(net_y_values_test(:, i, j), y_test_original);
    end
    
    % u�rednianie b��d�w �redniokwadratowych do wektora po liczbie iteracji
    net_mean_mse_train(:, i) = mean(net_mse_train(:, i, :));
    net_mean_mse_test(:, i) = mean(net_mse_test(:, i, :));
end

% wykresy u�rednionych b��d�w dla uczenia i testowania
figure(1);
plot(1:1:length(net_mean_mse_train), net_mean_mse_train, '-o', 1:1:length(net_mean_mse_test), net_mean_mse_test, '-o'); 
legend('trenowanie', ('testowanie'));
xlabel('liczba neuron�w', 'fontsize', 17);
ylabel('b��d', 'fontsize', 17);
grid on;

% zastosowanie algorytmu Kfold do skrajnej oceny krzy�owej
nr_of_iter = length(y_train);
indices = crossvalind('Kfold', y_train, nr_of_iter);
for i = 1:1:max_number_of_hidden_nurons
    for j = 1:1:nr_of_iter
        train = (indices ~= i);
 
        net = train_net(x_train(train, :), y_train(train, :), i);
        Z = calc_jacobian(net, x_train(train, :));
        rank_Z(i, j) = rank(Z);
        % ilo�� parametr�w = 1*liczba_neuron�w_ukrytych +
        % liczba_bias�w_dla_neuron�w_ukrytych + liczba_neuron�w_ukrytych*1
        % + liczba_bias�w_dla_wyj�cia_sieci
        q(i, j) = 3*net.layerweights{2,1}.size(2) + 1;
        if rank_Z(i, j) == q(i, j)
            [U, W, V] = svd(Z);
            H = Z*V*inv(W'*W)*V'*Z';
            h_kk{i, j} = diag(H);
            var_h_kk(i, j) = var(h_kk{i, j});
        else
            h_kk{i, j} = NaN;
            var_h_kk(i, j) = NaN;
        end
    end
end