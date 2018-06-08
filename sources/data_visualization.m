clear all;
close all;
clc;

% wizualizacja zbioru testowego
train_data = load('zestaw_apr_2_test.txt');
figure(1);
plot(train_data(:, 1), train_data(:, 2), 'o'); 
xlabel('x', 'FontSize', 17);
ylabel('f(x)', 'FontSize', 17);

% wizualizacja zbioru uczacego
train_data = load('zestaw_apr_2_train.txt');
figure(2);
plot(train_data(:, 1), train_data(:, 2), 'o'); 
xlabel('x' ,'FontSize', 17);
ylabel('f(x)', 'FontSize', 17);