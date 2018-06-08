function net = train_net(x_data, y_data, number_of_hidden_neurons)

    rand('state', sum(100*clock));
    
    net = newff(x_data', y_data', number_of_hidden_neurons, {'tansig', 'purelin'}, 'traingd');
    net.IW{1, 1} = (0.3*rand(number_of_hidden_neurons, 1)) - 0.15;
    net.b{1} = (0.3*rand(number_of_hidden_neurons, 1)) - 0.15;
    net.LW{2, 1} = (0.3*rand(1, number_of_hidden_neurons)) - 0.15;
    net.b{2} = (0.3*rand(1)) - 0.15;           
    net.trainParam.epochs = 100;           
    net.trainParam.showWindow = false; 
    net = train(net, x_data', y_data');
    net.trainFcn = 'trainlm';
    net.trainParam.epochs = 200;           
    net.trainParam.showWindow = false; 
    net = train(net, x_data', y_data');
end

