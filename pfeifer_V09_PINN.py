# -*- coding: utf-8 -*-
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

dirs = ['./model', './data', './figures']
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# upload the files to data folder

def main():

    scenario = 0 # 0: under-damped, 1: critically-damped, 2: over-damped

    # Load data
    base_filename = './data/pinn_data'

    # Determine the filename based on the scenario
    if scenario == 0:
        filename = base_filename + '_under-damped.pt'
    elif scenario == 1:
        filename = base_filename + '_critically-damped.pt'
    elif scenario == 2:
        filename = base_filename + '_over-damped.pt'
    else:
        raise ValueError("Invalid scenario value. Scenario should be 0, 1, or 2.")

    # Load the tensor from the file
    pinn_data = torch.load(filename)
    noise = np.random.normal(0, 0.25, len(pinn_data[:, 1]))

    for i in range(len(pinn_data[:,1])):
        pinn_data[i,1] = pinn_data[i,1]+noise[i]

    plt.plot(pinn_data[:,0], pinn_data[:,1], label='Mesaurement', marker='o', linestyle='None')
    plt.show()
    #exit()

    # Extract t_f and u_test_pred from the loaded tensor
    t_f, u_test_pred = pinn_data[:, 0], pinn_data[:, 1]

    # # If necessary, reshape them back to [N, 1]
    t_f = t_f.unsqueeze(1)
    u_test_pred = u_test_pred.unsqueeze(1)


    t_range = [0.0, 6.0]
    VC_0 = 12.0
    C, M, K = 0.1, 0.1, 0.1
    ode_parameters = [t_range, C, M, K, VC_0]

    # train parameters
    ni = 50
    optimizer = 0  # 0: L-BFGS 1: Adam 2: SGD
    max_epochs = 100
    min_loss = 1e-8
    learning_rate = 0.01
    train_parameters = [scenario, ni, optimizer, max_epochs, min_loss, learning_rate]

    # test parameters
    test_parameters = [scenario, ni]

    # Neural networks parameters
    nn_layers = [1, 128, 128, 1]  # neural networks layers
    act_fun = 'gelu'
    nn_parameters = [nn_layers, act_fun]

    train(ode_parameters, train_parameters, nn_parameters, t_f, u_test_pred)  # train the model
    # test(ode_parameters, test_parameters) # test the model


class NeuralNetwork(nn.Module):
    def __init__(self, parameters):
        super(NeuralNetwork, self).__init__()
        [nn_layers, act_fun] = parameters

        # Define a dictionary for activation functions
        af_list = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }
        self.activation_function = af_list.get(act_fun, None)

        # Check if activation function is provided
        if self.activation_function is None:
            raise ValueError(f"Activation function '{act_fun}' is not supported.")

        # Create layers dynamically based on nn_layers
        self.layers = nn.ModuleList()
        for i in range(len(nn_layers) - 1):
            self.layers.append(nn.Linear(nn_layers[i], nn_layers[i + 1]))
            if i < len(nn_layers) - 2:  # No activation function after the last layer
                self.layers.append(self.activation_function)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def pred(net_model, t):
    model_return = net_model(t)
    return model_return

def squared_difference(input, target):
    return (input - target) ** 2

def calculate_analytical_solution(ode_parameters, scenario, ni):
    t_range, _, _, _, VC_0 = ode_parameters

    i_calc = np.zeros(ni)
    t_test = np.linspace(t_range[0], t_range[1], ni)

    if scenario == 0:  # under-damped
        C = 1.2   # Daempfung
        M = 1.5   # Masse
        K = 3.3   # Steifigkeit
        #VC_0 = 12.0  # Initiale Geschwindigkeit
        VC_0 = 12.0
    elif scenario == 1:  # critically-damped
        C = 4.47  # Daempfung
        M = 1.5   # Masse
        K = 3.3   # Steifigkeit
        VC_0 = 12.0  # Initiale Geschwindigkeit
    elif scenario == 2:  # over-damped
        C = 6.0   # Daempfung
        M = 1.5   # Masse
        K = 3.3   # Steifigkeit
        VC_0 = 12.0  # volts, initial capacitor voltage
    else:
        raise ValueError("Invalid scenario value. Scenario should be 0 (under-damped), 1 (critically-damped), or 2 (over-damped).")

    alpha = C / (2 * M)

    for i in range(ni):
        if scenario == 0:  # under-damped
            i_calc[i] = 8.4 * np.exp(-0.4 * t_test[i]) * np.sin(1.44 * t_test[i])
        elif scenario == 1:  # critically-damped
            i_calc[i] = (VC_0 / M) * t_test[i] * np.exp(-1 * alpha * t_test[i])
        elif scenario == 2:  # over-damped
            i_calc[i] = 3.0 * (np.exp(-0.67 * t_test[i]) - np.exp(-3.33 * t_test[i]))
        else:
            raise ValueError("Invalid scenario value. Scenario should be 0 (under-damped), 1 (critically-damped), or 2 (over-damped).")

    return t_test, i_calc


def get_plot_title(scenario):
    if scenario == 0:  # under-damped
        return "Under-Damped Scenario"
    elif scenario == 1:  # critically-damped
        return "Critically-Damped Scenario"
    elif scenario == 2:  # over-damped
        return "Over-Damped Scenario"
    else:
        raise ValueError("Invalid scenario value. Scenario should be 0 (under-damped), 1 (critically-damped), or 2 (over-damped).")


def train(ode_parameters, train_parameters, nn_parameters, t_f, u_test_pred):
    # loading parameters
    [scenario, ni, opt, max_epochs, min_loss, learning_rate] = train_parameters
    [t_range, C, M, K, VC_0] = ode_parameters

    # initial condition - same as for PINN
    u_0 = [0.0];
    t_i = torch.FloatTensor(np.array(t_range[0]).reshape(-1, 1))
    u_i = torch.FloatTensor(np.array(u_0).reshape(-1, 1))

    #  f'(t) initial conditions - same as for PINN
    u_i2 = torch.FloatTensor(np.array(VC_0).reshape(-1, 1))

    t_i.requires_grad = True
    t_f.requires_grad = True


    net_model = NeuralNetwork(nn_parameters)

    train_C = torch.nn.Parameter(torch.FloatTensor([C]), requires_grad=True)
    net_model.register_parameter('C', train_C)

    train_M = torch.nn.Parameter(torch.FloatTensor([M]), requires_grad=True)
    net_model.register_parameter('M', train_M)

    train_K = torch.nn.Parameter(torch.FloatTensor([K]), requires_grad=True)
    net_model.register_parameter('K', train_K)

    # choose optimizer
    if opt == 1:
        optimizer = torch.optim.Adam([{'params': net_model.parameters()}], lr=learning_rate)
    elif opt == 2:
        optimizer = torch.optim.SGD([{'params': net_model.parameters()}], lr=learning_rate)
    else:
        optimizer = torch.optim.LBFGS([{'params': net_model.parameters()}], lr=learning_rate)


    epoch = 0
    loss = 10
    loss_record = np.empty([0, 7])
    rlc_record = np.empty([0, 3])
    plt.ion()
    print('------------------------Neural network------------------------------------')
    print(net_model)
    print('----------------------------Optimizer--------------------------------------')
    print(optimizer)
    #  -----------   start training   ------------
    starttime_train = datetime.datetime.now()
    formatted_time = starttime_train.strftime("%Y-%m-%d %H:%M:%S")
    print('------------------------Start training:{}---------------------'.format(formatted_time))

    while epoch < max_epochs and loss > min_loss:
        def compute_losses():

          # u_i refers to the initial condition, u_f to the solution
          u_i_pred = pred(net_model, t_i)
          u_f_pred = pred(net_model, t_f)

          u_i_pred_dt = torch.autograd.grad(u_i_pred.sum(), t_i, create_graph=True)[0]

          u_f_pred_dt = torch.autograd.grad(u_f_pred.sum(), t_f, create_graph=True)[0]
          u_f_pred_dtt = torch.autograd.grad(u_f_pred_dt.sum(), t_f, create_graph=True)[0]

          f = (train_M * u_f_pred_dtt) + (train_C * u_f_pred_dt) + (train_K * u_f_pred)

          # get the three loss components
          loss_1 = torch.mean(squared_difference(u_i_pred, u_i))
          loss_2 = torch.mean(squared_difference(f, torch.zeros_like(t_f)))
          loss_3 = torch.mean(squared_difference(u_i_pred_dt, (u_i2 / train_M)))
          loss_4 = torch.mean(squared_difference(u_f_pred, u_test_pred))


          return [loss_1, loss_2, loss_3, loss_4]

        def closure():
            # Compute individual loss terms
            loss_terms = compute_losses()

            # Calculate total loss
            loss_total = sum(loss_terms)

            optimizer.zero_grad()
            loss_total.backward()

            # Return total loss
            return loss_total

        optimizer.step(closure)

        # Compute and store losses after optimizer step
        loss_terms = compute_losses()
        loss_total = sum(loss_terms)
        loss_value = loss_total.item()

        # Store loss values
        step_time = datetime.datetime.now() - starttime_train
        loss_record = np.append(loss_record, [
            [epoch, step_time.seconds + step_time.microseconds / 1000000, loss_value] + [l.cpu().data.numpy() for l in
                                                                                         loss_terms]], axis=0)

        rlc_record = np.append(
            rlc_record,
            [[train_C.item(), train_M.item(), train_K.item()]],  # Convert to scalar values
            axis=0
        )

        if epoch % 100 == 0:
          print('Running: ', epoch, ' / ', max_epochs, '     loss: ', loss_value)

        if epoch == max_epochs - 1 or loss <= min_loss:
          print(f"Final values at epoch {epoch}: C = {train_C.item()}, M = {train_M.item()}, K = {train_K.item()}")


        epoch += 1

        _, i_calc = calculate_analytical_solution(ode_parameters, scenario, ni)


    endtime_train = datetime.datetime.now()
    print('---------------End training:{}---------------'.format(endtime_train))
    torch.save(net_model, './model/PINN.pkl')

    train_time = endtime_train - starttime_train
    print('---------------Training time:{}s---------------'.format(train_time.seconds + train_time.microseconds / 1e6))


    with torch.no_grad():
      trained_model = torch.load('./model/PINN.pkl')

      plot_title = get_plot_title(scenario)

      plt.clf()
      u_test_pred = pred(trained_model, t_f)
      plt.plot(t_f, u_test_pred, label='Predicted solution')
      plt.plot(t_f, i_calc, label='Analytical solution')
      plt.title(plot_title)
      plt.xlabel('Time')
      plt.ylabel('Displacement')
      plt.legend(loc='upper right')
      plt.pause(0.1)

      plt.ioff()
      plt.show()

      # Plot all loss terms as residual plots
      plt.figure(figsize=(10, 6))
      for i in range(4):
          plt.plot(loss_record[:, 0], loss_record[:, i + 3], label=f'Loss term {i + 1}')
      plt.title('Residual Plots for Loss Terms')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.ylim((0, 10))
      plt.legend()
      plt.show()

      plt.figure()
      plt.plot(loss_record[:, 0], loss_record[:, 2])
      plt.title('Loss value')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.show()

      # plot the last 100 values of loss
      plt.figure()
      plt.plot(loss_record[-100:, 0], loss_record[-100:, 2])
      plt.show()


def test(ode_parameters, test_parameters):
  [t_range, C, M, K, VC_0] = ode_parameters
  [scenario, ni] = test_parameters

  trained_model = torch.load('./model/PINN.pkl')

  t_test, i_calc = calculate_analytical_solution(ode_parameters, scenario, ni)

  t_test_tens = torch.FloatTensor(np.linspace(t_range[0], t_range[1], ni, endpoint=True).reshape(-1, 1))

  with torch.no_grad():
    trained_model = torch.load('./model/PINN.pkl')

    plot_title = get_plot_title(scenario)

    plt.clf()
    u_test_pred = pred(trained_model, t_test_tens)
    plt.plot(t_test_tens, u_test_pred, label='Predicted solution')
    plt.plot(t_test, i_calc, label='Analytical solution')

    plt.title(plot_title)
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend(loc='upper right')
    plt.pause(0.1)



if __name__ == '__main__':
    main()
    pass

