clear all;
close all;
clc;

% parametry modelu i dane treningowe
optimal_number_of_hidden_neurons = 7; % UWAGA - od pewnej liczby neuron�w wszystkie sicie mog� poisada� macierze Jacobiego niepe�nego rz�du - wtedy program zwr�ci b��d 
train_data = load('zestaw_apr_2_train.txt');
train_size = size(train_data, 1);
q = 3*optimal_number_of_hidden_neurons + 1;
nr_of_iter = 50;

% normalizacja danych treningowych
x_train_original = train_data(:, 1);
[x_train, x_train_max, x_train_min, x_train_mean] = regularize_data(x_train_original);
y_train_original = train_data(:, 2);
[y_train, y_train_max, y_train_min, y_train_mean] = regularize_data(y_train_original);

% inicjalizacja 
y_pred = ones(nr_of_iter, length(y_train))*NaN;
err = ones(nr_of_iter, length(y_train))*NaN;
h_kk = ones(nr_of_iter, length(y_train))*NaN;
Ep = ones(1, nr_of_iter)*NaN;
mi = ones(1,nr_of_iter)*NaN;

% budowa okre�lonej liczby sieci i zbieranie statystyk
for i = 1:1:nr_of_iter
    net = train_net(x_train, y_train, optimal_number_of_hidden_neurons);
    Z = calc_jacobian(net, x_train);
    rank_Z(i) = rank(Z);
    
    if rank_Z(i) == q
        nets{i} = net;
        y_pred(i, :) = deregularize_data(sim(net, x_train'), y_train_max, y_train_min, y_train_mean);
        err(i, :) = (y_pred(i, :) - y_train_original')';
        [U, W, V] = svd(Z);
        H = Z*V*inv(W'*W)*V'*Z'; 
        h_kk(i, :) = diag(H);
        Ep(i) = sqrt((1/train_size)*sum((err(i, :)./(1 - h_kk(i, :))).^2));
        mi(i) = 1/train_size*sum(sqrt(train_size/q*h_kk(i, :)));
    else
        nets{i} = NaN;
    end
end
    
% wykresy mi - Ep
figure(1);
plot(mi(~isnan(mi)), Ep(~isnan(Ep)), 'rx');
xlabel('\boldmath${\mu}$', 'Interpreter', 'latex', 'fontweight', 'bold', 'fontsize', 17);
ylabel('\boldmath${E_p}$', 'Interpreter', 'latex', 'fontweight', 'bold', 'fontsize', 17);
grid on;

% wyniki mi - Ep
disp('Wyniki mi - Ep');
Ep
mi

% poziom istotno�ci i funkcja kwantylowa t - Studenta
alpha = 0.95;
ta = tinv(alpha, train_size - q);

% wyb�r najlepszego modelu wzgl�dem metryki Euklidesa
model_nr = getModel(Ep, mi);

% estymator odchylenia standardowego
s = sqrt(sum(err(model_nr, :).^2)/(train_size - q));

% histogram d�wigni
figure(2);
hist(h_kk(model_nr, :), 20);
xlabel('warto�� wagi', 'fontsize', 17);
ylabel('liczba wag', 'fontsize', 17);
grid on;

% wybrana aproksymacja dla zbioru treningowego
figure(3);
plot(x_train_original, y_train_original, 'rx', x_train_original, y_pred(model_nr, :));
legend('oryginalne punkty', 'model');
xlabel('x', 'fontsize', 17);
ylabel('y', 'fontsize', 17);
grid on;

% przedzia� ufno�ci pomiaru wyj�cia sieci neuronowej na poziomie 1 - alpha
y_lower_bound_1 = y_pred(model_nr, :) - ta*s;
y_higher_bound_1 = y_pred(model_nr, :) + ta*s;
figure(4);
plot(x_train_original, y_train_original, 'rx', x_train_original, y_pred(model_nr, :), x_train_original, y_lower_bound_1, ...
'g', x_train_original, y_higher_bound_1, 'y');
legend('oryginale punkty', 'model', 'dolna granica modelu', 'g�rna granica modelu');
xlabel('x', 'fontsize', 17);
ylabel('y', 'fontsize', 17);
grid on;

% Przedzia� ufno�ci wielko�ci wyj�ciowej sieci neuronowej dla k-tego przyk�adu ze zbioru
% ucz�cego na poziomie 1 - alpha
y_lower_bound_2 = y_pred(model_nr, :) - ta*s*sqrt(h_kk(model_nr, :));
y_higher_bound_2 = y_pred(model_nr, :) + ta*s*sqrt(h_kk(model_nr, :));
figure(5);
plot(x_train_original, y_train_original, 'rx', x_train_original, y_pred(model_nr, :), x_train_original, y_lower_bound_2, ...
'g', x_train_original, y_higher_bound_2, 'y');
legend('oryginale punkty', 'model', 'dolna granica modelu', 'g�rna granica modelu');
xlabel('x', 'fontsize', 17);
ylabel('y', 'fontsize', 17);
grid on;

% Przedzia� ufno�ci predykcji dla k-tego przyk�adu ze zbioru ucz�cego na
% poziomie 1 - alpha
y_lower_bound_3 = y_pred(model_nr, :) - ta*s*sqrt(h_kk(model_nr, :)./(1. - h_kk(model_nr, :)));
y_higher_bound_3 = y_pred(model_nr, :) + ta*s*sqrt(h_kk(model_nr, :)./(1. - h_kk(model_nr, :)));
figure(6);
plot(x_train_original, y_train_original, 'rx', x_train_original, y_pred(model_nr, :), x_train_original, y_lower_bound_3, ...
'g', x_train_original, y_higher_bound_3, 'y');
legend('oryginale punkty', 'model', 'dolna granica modelu', 'g�rna granica modelu');
xlabel('x', 'fontsize', 17);
ylabel('y', 'fontsize', 17);
grid on;

% zapis wag do pliku
file = fopen('weights.txt','w');
fprintf(file, '%s \n %f \n', 'b_1' , nets{model_nr}.b{1});
fprintf(file, '\n');
fprintf(file, '%s \n %f \n', 'w_1' , nets{model_nr}.IW{1,1});
fprintf(file, '\n');
fprintf(file, '%s \n %f \n', 'b_2' , nets{model_nr}.b{2});
fprintf(file, '\n');
fprintf(file, '%s \n %f \n', 'w_2' , nets{model_nr}.LW{2,1});
fclose(file);

% sprawdzenie sieci dla zbioru testowego
test_data = load('zestaw_apr_2_test.txt');
test_size = size(test_data, 1);
x_test_original = test_data(:, 1);
x_test = (x_test_original - x_train_mean)./(x_train_max - x_train_min);
y_test_original = test_data(:, 2);
y_test_pred = deregularize_data(sim(nets{model_nr}, x_test'), y_train_max, y_train_min, y_train_mean);
figure(7);
plot(x_test_original, y_test_original, 'rx', x_test_original, y_test_pred);
legend('oryginalne punkty', 'model');
xlabel('x', 'fontsize', 17);
ylabel('y', 'fontsize', 17);
grid on;