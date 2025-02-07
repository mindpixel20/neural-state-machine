import nsm as n
import random as r
import time

def gen_bin(amt):
    bits = []
    for i in range(0, amt):
        rn = r.random()
        if rn > 0.5:
            bits.append(1)
        else:
            bits.append(0)
    return bits

length_in = 256
length_feedback = 256
amt = 500
training_data = []
print(f"Generating training data with {amt} elements of {length_in} bits for input and {length_feedback} bits for feedback")

for i in range(0, amt):
    in_bits = gen_bin(length_in + length_feedback)
    out_bits = gen_bin(length_in + length_feedback)
    training_data.append((in_bits, out_bits))

print("Initializing neural state machine with data")

nsm = n.neural_state_machine(length_in,4,length_feedback, training_data, True)

amt_to_test = 500

print(f"Iterating through dataset of {amt_to_test} data points.")
start_time = time.time()
proof = []
for i in range(0, amt_to_test):
    t_in = training_data[i%amt][0][0:length_in]
    #print(t_in, type(t_in))
    proof.append(nsm.get_z(t_in))
end_time = time.time()
total = end_time-start_time
print(f"{amt_to_test} rounds of inference complete in {total} seconds.")
print(f"Proof: {len(proof)}")

n.save_nsm_model("test", nsm)
print("done")
