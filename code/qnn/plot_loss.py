import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# BFGS

def plotAndShow(_title):
    fig, ax = plt.subplots()
    plt.title(_title)
    plt.ylabel("Objective function value")
    plt.xlabel("Iterations")
    for index, item in enumerate(_obj_func_eval):
        ax.plot(item, label='layer ' + str(pow(2, index)))
    ax.legend()
    plt.show()


#######################################################################################################################################################
# adhoc_20 circuit 01
_obj_func_eval = [
    # Layer 1
    np.array([41.5986328125, 37.9375, 37.2626953125, 35.5322265625, 35.3740234375, 35.0, 35.078125, 35.14453125, 35.16015625, 35.09375, 35.1552734375]),
    # Layer 2
    np.array([38.9580078125, 40.2021484375, 38.111328125, 37.6025390625, 38.6962890625, 34.9970703125, 35.1484375, 34.6201171875, 34.4375, 33.89453125,
             34.5986328125, 33.634765625, 33.359375, 35.244140625, 33.5595703125, 33.6826171875, 33.6640625, 33.515625, 33.8115234375, 33.666015625]),
    # Layer 4
    np.array([40.9404296875, 38.0859375, 36.435546875, 42.796875, 35.6904296875, 35.16015625, 34.97265625, 34.1708984375, 33.8544921875, 33.4951171875,
             33.33984375, 33.41015625, 33.3935546875, 33.51953125, 33.478515625, 33.5107421875, 33.3359375, 33.5107421875, 33.5244140625]),
    # Layer 8
    np.array([39.9189453125, 36.0859375, 36.703125, 35.146484375, 34.4833984375, 33.603515625, 33.177734375, 33.0283203125,
             32.8115234375, 33.1591796875, 32.916015625, 32.8388671875, 33.1328125, 33.0244140625, 32.99609375]),
]

#plotAndShow('Adhoc: circuit_qiskit_01 / BFGS optimizer')

#######################################################################################################################################################
# custom_0 circuit 01
_obj_func_eval = [
    # Layer 1
    np.array([41.5126953125, 39.333984375, 41.322265625, 38.630859375, 39.189453125, 40.2021484375, 39.193359375, 38.8701171875, 40.419921875, 39.8447265625,
             38.7744140625, 38.546875, 39.857421875, 38.236328125, 38.5869140625, 38.44921875, 38.7109375, 38.634765625, 38.560546875, 38.439453125]),
    # Layer 2
    np.array([41.421875, 40.2236328125, 40.0205078125, 39.3271484375, 38.7392578125, 37.6904296875, 38.21484375, 37.8427734375, 37.70703125, 37.9873046875, 37.771484375, 37.9267578125, 37.76953125]),
    # Layer 4
    np.array([38.9072265625, 39.2705078125, 38.638671875, 38.1806640625, 37.8310546875, 37.7880859375, 37.9423828125, 37.8828125, 37.6875, 37.9365234375,
             37.806640625, 37.78515625, 37.7607421875, 37.5703125, 37.705078125, 37.6845703125, 37.611328125, 37.8330078125, 37.97265625, 37.9794921875]),
    # Layer 8
    np.array([41.1845703125, 39.9287109375, 39.072265625, 40.650390625, 38.6669921875, 38.037109375, 38.2724609375, 37.9248046875,
             37.822265625, 37.4482421875, 37.8818359375, 37.5712890625, 37.568359375, 37.7646484375, 37.513671875]),
]

# plotAndShow('Custom: circuit_qiskit_01 / BFGS optimizer')

#######################################################################################################################################################
# iris_10 circuit 01
_obj_func_eval = [
    # Layer 1
    np.array([45.1328125, 25.9580078125, 26.2490234375, 22.287109375, 22.1318359375, 20.33984375, 28.2958984375, 19.828125, 18.62109375, 18.1328125, 17.62890625, 17.03515625, 19.2783203125,
              16.1513671875, 15.19921875, 15.19140625, 14.3515625, 14.01953125, 13.77734375, 13.78515625, 13.71875, 13.6943359375, 13.607421875, 13.6884765625, 13.6669921875, 13.7099609375,
              13.4736328125, 13.71484375, 13.3916015625, 13.533203125, 13.4384765625, 13.287109375, 13.39453125, 13.580078125, 13.638671875, 13.4619140625, 13.458984375]),
    # Layer 2
    np.array([27.984375, 22.44921875, 17.9248046875, 17.126953125, 16.3408203125, 15.7890625, 15.046875, 14.8427734375, 13.8212890625, 12.8125, 11.3662109375, 10.8203125, 10.5322265625, 10.5048828125,
              10.1396484375, 9.8984375, 9.5712890625, 9.4931640625, 9.00390625, 8.6689453125, 8.1943359375, 8.046875, 7.802734375, 7.6875, 7.544921875, 7.359375, 7.5146484375, 7.537109375,
              7.509765625, 8.1484375, 7.6220703125, 7.3701171875, 7.5888671875, 7.4873046875, 7.6259765625, 7.63671875, 7.4990234375]),
    # Layer 4
    np.array([40.052734375, 48.5830078125, 37.1396484375, 35.7939453125, 34.009765625, 33.716796875, 26.7724609375, 22.6357421875, 15.328125, 12.099609375, 11.90625, 11.0537109375,
             10.423828125, 9.9921875, 9.533203125, 9.19140625, 8.8408203125, 8.6435546875, 8.4130859375, 8.6005859375, 8.4794921875, 8.443359375, 8.53125, 8.494140625, 8.443359375]),
    # Layer 8
    np.array([52.2431640625, 39.3447265625, 26.39453125, 36.5791015625, 23.078125, 20.1357421875, 17.4033203125, 14.1416015625, 11.8056640625, 10.9345703125, 10.39453125, 9.9931640625, 9.0126953125,
              8.5283203125, 8.255859375, 7.9296875, 7.4951171875, 8.134765625, 7.46484375, 7.30078125, 7.31640625, 7.2529296875, 7.1943359375, 7.1162109375, 7.0859375, 7.0009765625, 6.9599609375,
              6.97265625, 6.9765625, 6.9716796875, 6.8916015625, 7.03515625, 7.1474609375, 7.0693359375, 6.90625, 7.099609375]),
]

# plotAndShow('Iris: circuit_qiskit_01 / BFGS optimizer')

#######################################################################################################################################################
# rain_30 circuit 01
_obj_func_eval = [
    # Layer 1
    np.array([38.7109375, 36.9248046875, 37.02734375, 36.88671875, 36.2666015625, 36.20703125, 35.6484375, 43.240234375, 35.458984375, 38.4365234375, 39.3681640625, 36.8818359375, 35.4150390625,
              36.4736328125, 35.2919921875, 35.2119140625, 34.7890625, 34.748046875, 34.5654296875, 34.6416015625, 34.3642578125, 34.4072265625, 34.4697265625, 34.330078125, 34.60546875,
              34.6123046875, 34.5849609375, 34.2763671875, 34.5751953125, 34.3359375, 34.6025390625, 34.373046875]),
    # Layer 2
    np.array([39.7646484375, 38.890625, 38.5458984375, 39.3212890625, 38.0966796875, 38.2470703125, 37.6044921875, 37.23828125, 35.66796875, 35.4423828125,
             35.1298828125, 35.2919921875, 35.0966796875, 34.8642578125, 35.052734375, 35.2177734375, 35.181640625, 35.0693359375, 35.0830078125]),
    # Layer 4
    np.array([41.662109375, 38.4306640625, 37.505859375, 37.26953125, 36.625, 36.71484375, 36.6611328125, 36.8037109375,
             36.7890625, 36.564453125, 36.6044921875, 36.9267578125, 36.5556640625, 36.953125, 36.90625, 36.765625]),
    # Layer 8
    np.array([39.4267578125, 38.92578125, 39.3857421875, 38.107421875, 38.3291015625, 38.2490234375, 37.9580078125, 37.77734375, 37.0703125, 42.3154296875, 35.58203125, 41.1767578125,
             35.1552734375, 34.619140625, 34.4228515625, 34.1259765625, 34.0615234375, 33.6845703125, 33.9150390625, 33.919921875, 33.734375, 33.9140625, 33.9228515625]),
]

# plotAndShow('Rain: circuit_qiskit_01 / BFGS optimizer')

#######################################################################################################################################################
# vlds_40 circuit 01
_obj_func_eval = [
    # Layer 1
    np.array([41.7431640625, 37.181640625, 36.056640625, 34.21484375, 34.927734375, 32.7421875, 30.7587890625, 30.1201171875, 28.2119140625, 28.013671875, 27.640625, 27.599609375, 27.05859375,
              27.0546875, 26.8134765625, 26.22265625, 26.1884765625, 25.83203125, 25.689453125, 25.64453125, 25.5419921875, 26.1435546875, 25.453125, 25.30859375, 25.1337890625, 25.1220703125,
              24.8740234375, 24.9169921875, 25.15625, 25.0146484375, 24.9326171875, 25.005859375, 25.0546875]),
    # Layer 2
    np.array([39.4072265625, 36.1015625, 32.6689453125, 30.7353515625, 28.9599609375, 28.1787109375, 26.837890625, 26.017578125,
             25.853515625, 25.87109375, 25.6005859375, 25.6396484375, 25.9365234375, 25.96484375, 25.85546875, 25.6884765625, 25.9404296875]),
    # Layer 4
    np.array([33.9990234375, 42.875, 30.3896484375, 30.2236328125, 29.74609375, 29.0244140625, 28.044921875, 27.3505859375,
             26.9052734375, 27.2265625, 27.0166015625, 27.0908203125, 27.1015625, 26.931640625, 27.2060546875]),
    # Layer 8
    np.array([34.5029296875, 47.28515625, 30.970703125, 30.4130859375, 29.484375, 28.8232421875, 28.044921875, 26.625, 26.623046875, 26.5517578125, 26.1923828125,
             26.0224609375, 25.6748046875, 25.2294921875, 25.02734375, 25.0888671875, 25.1220703125, 25.177734375, 25.158203125, 25.20703125, 25.1591796875]),
]

plotAndShow('Vlds: circuit_qiskit_01 / BFGS optimizer')