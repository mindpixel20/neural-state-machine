import random as r
import time
import pickle

class strutil:
    def __init__(self):
        pass 
    def bit2str(self, bit):
        if bit == 0:
            return "0"
        elif bit == 1:
            return "1"

    def str2bit(self, strbit):
        if strbit == "0":
            return 0
        elif strbit == "1":
            return 1 

    def bin2str(self, bins):
        returnstr = ""
        for b in bins:
            returnstr = returnstr + self.bit2str(b)
        return returnstr

    def str2bin(self, binstr):
        bits = [] 
        for b in binstr:
            bits.append(self.str2bit(b))
        return bits

class coeff_cell:
    def __init__(self):
        self.t0 = 0.0
        self.t1 = 0.0
        self.ak0 = 0.0
        self.ak1 = 0.0

        self.pi = 0.0

    def update(self, training_element): # t_in is a vector, t_out is a bit
        t_in = training_element[0]
        t_out = training_element[1]
        if t_out == 0:
            self.t0 += 1.0
        elif t_out == 1:
            self.t1 += 1.0

        if t_in == 1 and t_out == 0:
            self.ak0 += 1.0
        elif t_in == 1 and t_out == 1:
            self.ak1 += 1.0

        if self.t0 == 0.0 or self.t1 == 0.0:
            self.pi = 1.0
            return 1.0
        else:
            self.pi = abs((self.ak1/self.t1)-(self.ak0/self.t0))
            return self.pi 

class mem_cell:
    def __init__(self, i, z):
        self.i = i
        self.z = z

class nsm_neuron:
    def __init__(self, training_data=[], pi_cell_length=0):
        self.memory = {}
        self.pi = []
        self.u = strutil()
        self.last_output = 0

        self.pi_cells = []

        if training_data:
            self.batch_train(training_data)

        if pi_cell_length > 0:
            for i in range(0, pi_cell_length):
                self.pi_cells.append(coeff_cell())
                self.pi.append(0.0)
                
    def calc_disc_coeff(self, k, training_set):
        # k indexes which input this will be calculated for 
        t0 = 0.0 # number of outputs that are 0
        t1 = 0.0 # number of outputs that are 1 
        ak1 = 0.0 # number of times ik = 1 when zj = 1
        ak0 = 0.0 # number of times ik = 1 when zj = 0

        for t in training_set:
            if t[1] == 0:
                t0 += 1.0
            elif t[1] == 1:
                t1 += 1.0

        #for t in training_set:
            if t[0][k] == 1 and t[1] == 1:
                ak1 += 1.0
            elif t[0][k] == 1 and t[1] == 0:
                ak0 += 1.0
        if t0 == 0.0 or t1 == 0.0:
            return 1.0
        else:
            return abs((ak1/t1)-(ak0/t0))

    def coeff_vector_old(self, training_set):
        coeffs = []
        for i in range(0, len(training_set[0][0])):
            coeffs.append(self.calc_disc_coeff(i, training_set))
        return coeffs

    def coeff_vector(self, training_element):
        for i in range (0, len(self.pi_cells)):
            new_pi = self.pi_cells[i].update(training_element)
            self.pi[i] = new_pi  

    def disc_dist(self, x, y):
        if len(x) != len(y):
            return None
        if len(self.pi) == 0:
            print("PANIK")
            
        k = len(x) 
        summ = 0.0
        
        for i in range(0, k):
            summ += self.pi[i] * (abs(x[i] - y[i]))
        return summ

    def ham_dist(self, x, y): # really simple integer based hamming
        summ = 0
        if len(x) != len(y):
            return None

        for i in range(0, len(x)):
            if x[i] != y[i]:
                summ += 1
        return summ

    def train(self, x, y, pi_update=False):
        x_str = self.u.bin2str(x)
        if x_str not in self.memory:
            self.memory[x_str] = mem_cell(x, y)
        else:
            self.memory[x_str].z = y

        if pi_update and len(self.pi_cells) > 0:
            self.coeff_vector((x, y))
            return

        elif pi_update:
            training_data = []
            training_data.append((x,y))
            for m_key in self.memory:
                m = self.memory[m_key] 
                training_data.append((m.i, m.z))
            self.pi = self.coeff_vector(training_data)

    def batch_train(self, training_data, pi_update=False): # a list of tuples of [inputs],output
        for t in training_data:
            self.train(t[0], t[1], pi_update)
        self.pi = self.coeff_vector(training_data)

    def cga(self, input_vector, pre_ham=[]):
        ivstr = self.u.bin2str(input_vector)
        if ivstr in self.memory:
            return self.memory[ivstr].z
        else:
            bit = self.cga1(input_vector)
            if bit:
                self.last_output = bit
                return bit
            else:
                if(pre_ham):
                    bit = self.cga2(input_vector, pre_ham)
                if bit:
                    self.last_output = bit
                    return bit
                else: 
                    bit = self.cga3(input_vector)
                    self.last_output = bit
                    return bit
                

    def cga1(self, input_vector):
        # first, find the lowest disc_dist
        i_min = 66666666.0 # some arbirarily large number to start with
        cells = self.memory.values()
        bits = []
        for c in cells:
            tmp = self.disc_dist(c.i, input_vector)
            if tmp < i_min:
                bits = []
                i_min = tmp
                bits.append(c.z)
            elif tmp == i_min:
                bits.append(c.z) 
        # then build a list of those that have dd = i_min
        #bits = []
        #for c in cells:
            #tmp = self.disc_dist(c.i, input_vector)
            #if tmp == i_min:
                #bits.append(c.z)

        return self.bits_equal(bits) 

    def cga2(self, input_vector, precomputed_hamming):
        bits = []
        h_min = len(input_vector) # again, some gigantic starting number 
        if precomputed_hamming:
            for p in precomputed_hamming:
                bits.append(self.memory[p].z)
            return self.bits_equal(bits)
    
        else:
            x = x / 0
            # first, find the lowest hamming distance
            cells = self.memory.values()
            for c in cells:
                tmp = self.ham_dist(c.i, input_vector)
                if tmp < h_min:
                    h_min = tmp
            # then build a list of those that have hd = h_max
            cells = self.memory.values()
            for c in cells:
                tmp = self.ham_dist(c.i, input_vector)
                if tmp == h_min:
                    bits.append(c.z)
            return self.bits_equal(bits) 

    def cga3(self, input_vector):
        r_val = r.random()
        if r_val > 0.5:
            return 1
        else:
            return 0 

    def bits_equal(self, bits): # it'll return the bit if they're all the same!
        if len(bits) == 1: # needs to be two or more bits to check I think? 
            return None
        x = bits[0]
        for b in bits:
            if b != x:
                return None
        return x

class neural_state_machine:
    def __init__(self, k, l, m, initial_training_data=[],debug=False):
        self.k = k # neuron input width
        self.l = l # neuron output width
        self.m = m # neuron feedback width
        self.n = k + m # total neurons

        self.neurons = [] #

        for i in initial_training_data: # to pad and account for feedback if it isn't present
            if len(i[0]) < (k + m):
                while len(i[0]) < (k + m):
                    i[0].append(0)

        for i in range(0, self.n):
            self.neurons.append(nsm_neuron([], self.n))

        if initial_training_data:
            self.train_all(initial_training_data)

        self.u = strutil()

        self.debug = debug

        if self.debug:
            print("Neural state machine ready. Total neurons:",len(self.neurons))

    def fix_input(self, input_str): # wouldn't make sense to convert back if it'll need to be converted when processing
        bits = self.u.str2bin(input_str)
        if len(bits) > self.k:
            bits = bits[0:self.k]
            return bits
        elif len(bits) < self.k:
            while len(bits) < self.k:
                bits.append(0)
            return bits
        else:
            return bits

    def train_all(self, training_set):
        ctr = 1
        start = time.time()
        end = time.time()
        total = end - start
        total_time = total
        
        for t in training_set:
            start = time.time() 
            for i in range(0, self.n):
                self.neurons[i].train(t[0], t[1][i], True)
            end = time.time()
            total = end - start
            total_time += total
            #print(f"Training element {ctr} completed in {total} seconds. Total training time: {total_time} seconds")
            ctr += 1

        print(f"Total training time: {total_time} seconds")

    def train_io_only(self, training_set):
        for t in training_set:
            for i in range(0, self.k):
                self.neurons[i].train(t[0], t[1][i], True)

    def train_feedback_only(self, training_set):
        for t in training_set:
            for i in range(self.k, self.n):
                self.neurons[i].train(t[0], t[1][i], True)

    # assuming a fully connected network for simplicity
    def get_z(self, bits):
        for i in range(self.l, (self.l + self.m)): # feedback
            bits.append(self.neurons[i].last_output)

        # pre compute the hamming distances
        # first, get a neuron's memory to work with

        
        mem = self.neurons[0].memory
        bitstr = self.u.bin2str(bits)
        ham = [] # this will end up being a list of memory locations that're close enough to the input vector
        ham_dist = len(bits)
        if not bitstr in mem: 
            for mc in mem: # find the minimum distance
                tmp = self.hamming(mem[mc].i, bits)
                if tmp < ham_dist:
                    ham_dist = tmp
                    # clear the ham list
                    # then add that mc to the list
                    ham = []
                    ham.append(mc) 

                elif tmp == ham_dist:
                    ham.append(mc)
                    
            #if ham_dist < len(bits):
                #for mc in mem: # if found, find memory locations that're equal to or closer than the minimum distance
                    #tmp = self.hamming(mem[mc].i, bits)
                    #if tmp <= ham_dist:
                        #ham.append(mc) 
        
        outputs = []
        for i in range(0, self.l):
            bit = self.neurons[i].cga(bits, ham)
            outputs.append(bit)

        return outputs

    def get_z_str(self, input_vector):
        bits = self.fix_input(input_vector)
        outputs = self.get_z(bits)
        return self.u.bin2str(outputs[0:self.l])

    def hamming(self, x, y):
        if len(x) != len(y):
            return None
        summ = 0
        for i in range(len(x)):
            if x[i] != y[i]:
                summ += 1
        return summ

def save_nsm_model(filename, model):
    model_file = open(filename, 'ab')
    pickle.dump(model, model_file)
    model_file.close() 

def load_nsm_model(filename):
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model

        
        
#training_set = []
#training_set.append(([1,1,0,0],[1,1,1,1,0,0,0,0]))
#training_set.append(([0,1,1,1],[1,1,1,1,0,0,0,0]))
#training_set.append(([0,0,1,0],[0,0,0,0,0,0,0,0]))
#training_set.append(([0,1,0,1],[1,1,0,0,0,0,0,0]))

#nm = neural_state_machine(4,4,4, training_set)
#outs = nm.get_z_str("0101")
