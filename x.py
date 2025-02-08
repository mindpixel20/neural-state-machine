import nsm as n

training_data = []

training_data.append(([0,0,0,0],[1]))
training_data.append(([1,1,1,1],[1]))
training_data.append(([0,1,1,1],[0]))
training_data.append(([1,0,0,0],[0]))
training_data.append(([1,1,0,0],[1]))
training_data.append(([0,0,1,0],[0]))
training_data.append(([1,0,1,0],[0]))

nsm = n.neural_state_machine(4,1,4, training_data, True)
nsm.train_io_only([([0,0,0,0],[0])])
print(nsm.get_z(training_data[0][0]))
print(nsm.get_z(training_data[3][0]))
print(nsm.get_z(training_data[6][0]))
