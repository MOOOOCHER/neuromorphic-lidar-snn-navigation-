#!/usr/bin/env python
import numpy as np
# import pylab
import parameters as p
import nengo
import nengo_loihi
from nengo.utils.ensemble import tuning_curves
import time


class NengoLoihiSpikingNeuralNetwork():
    def __init__(self, w_l, w_r):
        # NEST options
        np.set_printoptions(precision=1)
        self.num_spikesL = 0
        self.num_spikesR = 0
        self.dt = 0.002
        self.w_l, self.w_r = self.setWeights(w_l, w_r)
        self.input_data = np.zeros(32)
        # create the network beforehand and modify the firing rates of the spike generators in each step
        with nengo.Network(seed=0) as self.network:
            # Create motor IAF neurons
            self.neuron_postLeft = nengo.Ensemble(label="left motor neuron", n_neurons=1, dimensions=1, intercepts=[0], max_rates=[300], encoders=nengo.dists.Choice([[0.3]]),
                                             neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Choice(np.array([0]))}))    #initial voltage has to be zero because of nengo_loihi
            self.neuron_postRight = nengo.Ensemble(label="right motor neuron", n_neurons=1, dimensions=1, intercepts=[0],max_rates=[300], encoders=nengo.dists.Choice([[0.3]]),
                                              neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Choice(np.array([0]))}))
            self.input_nodes = []
            self.spike_generators = []
            self.generators_probe = []
            self.connections_l = []
            self.connections_r = []
            self.tuningCurves = []
            # Create spike generator neurons with their respective input nodes
            for i in range(32):
              self.createSpikeGenerator(i)
            # Create Output spike detector
            self.spike_detectorLeft = nengo.Probe(self.neuron_postLeft.neurons)
            self.spike_detectorRight = nengo.Probe(self.neuron_postRight.neurons)

        self.sim = nengo_loihi.Simulator(self.network, dt=self.dt)
        # get turning curves of neurons and interpolate the corresponding inverse function
        self.tuningCurves.append(tuning_curves(self.spike_generators[0], self.sim))
        firingRates = self.tuningCurves[0][1][24:].reshape(26)
        inputValues = self.tuningCurves[0][0][24:].reshape(26)
        inputValues[0] = 0  #intercept of spike generator neuron are set to zero
        self.interpolation_function = np.poly1d(np.polyfit(firingRates,inputValues,deg=4))


    def simulate(self, dvs_data, reward):
        # Set poisson neuron firing frequency
        dvs_data = dvs_data.reshape(dvs_data.size)
        self.setInputData(dvs_data)
        # Simulate network
        self.sim.run(0.05)

        # Get left and right output spikes
        n_l = np.sum(self.sim.data[self.spike_detectorLeft]>0, axis=0)
        n_r = np.sum(self.sim.data[self.spike_detectorRight]>0, axis=0)
        # Compute the difference, because all the spikes from the step before are also counted
        retL = n_l[0] - self.num_spikesL
        retR = n_r[0] - self.num_spikesR
        # update the spike count
        self.num_spikesL = n_l[0]
        self.num_spikesR = n_r[0]
        #self.sim.reset(0) #Reset function isn't implemented yet in nengo_loihi
        # return num spikes
        return retL,  retR

    def createSpikeGenerator(self, i):
        input_node = nengo.Node(output=lambda t: self.input_data[i])
        gen = nengo.Ensemble(label="spike generator" + str(i), n_neurons=1, max_rates=nengo.dists.Choice(np.array([300])),
                             dimensions=1, encoders=nengo.dists.Choice([[1]]), intercepts=[0],
                             normalize_encoders=False,
                             neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Choice(np.array([0]))}))
        nengo.Connection(input_node, gen)
        # Create connection handles for left and right motor neuron
        conn_l = nengo.Connection(gen.neurons, self.neuron_postLeft.neurons, transform=self.w_l[0][i])
        conn_r = nengo.Connection(gen.neurons, self.neuron_postRight.neurons, transform=self.w_r[0][i])
        # Create Probes
        self.generators_probe.append(nengo.Probe(gen.neurons))
        self.input_nodes.append(input_node)
        self.spike_generators.append(gen)
        self.connections_l.append(conn_l)
        self.connections_r.append(conn_r)

    def setWeights(self, weights_l, weights_r):
        factor = 630    #dt=0.001 und factor 250; dt0.002 und factor 600-650 mit v_max = 0.39
        w_l = []
        for w in weights_l.reshape(weights_l.size):
            if w==0:    # for weight = 0 there is a problem in nengo when building the network
                w +=0.00001
            w_l.append((w*self.dt)/factor) # because when building the network the weight is divided by dt (=0,001)
            # we need the factor to see, for which weights the network works best, since nengo doesn't work exactly like nest
        w_r = []
        for w in weights_r.reshape(weights_r.size):
            if w==0:
                w +=0.00001 # for weight = 0 there is a problem in nengo when building the network

            w_r.append((w*self.dt)/factor)
        return np.array(w_l).reshape(1,32), np.array(w_r).reshape(1,32)
    def setInputData(self,arr):
        for i in range(len(self.input_data)):
            if arr[i] != 0:
                # tuning curves for each neuron is a tuple of two elements: one for output value corresponding with the firing rate
                if(len(self.tuningCurves)>0):
                    rate = arr[i] / p.max_spikes
                    rate = np.clip(rate, 0, 1) * p.max_poisson_freq
                    if rate >= 300:  # if bigger than max_rates
                        self.input_data[i] = 1  # for output = 1 the firing Rate is biggest, for output > 1 it wont get bigger
                    else:
                        self.input_data[i] = self.interpolation_function(rate)
                    continue
            self.input_data[i] = -1



class NengoLoihiSpikingNeuralNetworkNew():
    def __init__(self, w_l, w_r):
        # NEST options
        np.set_printoptions(precision=1)
        self.num_spikesL = 0
        self.num_spikesR = 0
        self.w_l, self.w_r = self.setWeights(w_l, w_r)
        self.input_data = np.zeros(32)
        # create the network beforehand and modify the firing rates of the spike generators in each step
        with nengo.Network(seed=0) as self.network:
            # Create motor IAF neurons
            self.neuron_postLeft = nengo.Ensemble(label="left motor neuron", n_neurons=1, dimensions=1, intercepts=[0], max_rates=[300], encoders=nengo.dists.Choice([[0.3]]),
                                             neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Choice(np.array([0]))}))    #initial voltage has to be zero because of nengo_loihi
            self.neuron_postRight = nengo.Ensemble(label="right motor neuron", n_neurons=1, dimensions=1, intercepts=[0],max_rates=[300], encoders=nengo.dists.Choice([[0.3]]),
                                              neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Choice(np.array([0]))}))
            self.input_nodes = nengo.Node(size_out=32,output=lambda t: self.input_data)
            self.spike_generators = []
            self.generators_probe = []
            self.connections_l = []
            self.connections_r = []
            self.tuningCurves = []
            # Create spike generator neurons with their respective input nodes
            for i in range(32):
              self.createSpikeGenerator(i)
            # Create Output spike detector
            self.spike_detectorLeft = nengo.Probe(self.neuron_postLeft.neurons)
            self.spike_detectorRight = nengo.Probe(self.neuron_postRight.neurons)

        self.sim = nengo_loihi.Simulator(self.network)


        # get turning curves of neurons and interpolate the corresponding inverse function
        self.tuningCurves.append(tuning_curves(self.spike_generators[0], self.sim))
        firingRates = self.tuningCurves[0][1][24:].reshape(26)
        inputValues = self.tuningCurves[0][0][24:].reshape(26)
        inputValues[0] = 0  #intercept of spike generator neuron are set to zero
        self.interpolation_function = np.poly1d(np.polyfit(firingRates,inputValues,deg=4))


    def simulate(self, dvs_data, reward):
        # Set poisson neuron firing frequency
        dvs_data = dvs_data.reshape(dvs_data.size)
        self.setInputData(dvs_data)
        # Simulate network
        self.sim.run(0.05)
        # Get left and right output spikes
        n_l = np.sum(self.sim.data[self.spike_detectorLeft]>0, axis=0)
        n_r = np.sum(self.sim.data[self.spike_detectorRight]>0, axis=0)
        # Compute the difference, because all the spikes from the step before are also counted
        retL = n_l[0] - self.num_spikesL
        retR = n_r[0] - self.num_spikesR
        # update the spike count
        self.num_spikesL = n_l[0]
        self.num_spikesR = n_r[0]
        ensemble_probes = []
        for i in range(32):
            probe = np.sum(self.sim.data[self.generators_probe[i]]>0, axis=0)
            ensemble_probes.append(probe)
        ensemble_probes = np.array(ensemble_probes).reshape(8,4)
        #self.sim.reset(0) #Reset function isn't implemented yet in nengo_loihi
        # return num spikes
        return retL,  retR

    def createSpikeGenerator(self, i):
        gen = nengo.Ensemble(label="spike generator" + str(i), n_neurons=1, max_rates=nengo.dists.Choice(np.array([300])),
                             dimensions=1, encoders=nengo.dists.Choice([[1]]), intercepts=[0],
                             normalize_encoders=False,
                             neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Choice(np.array([0]))}))
        nengo.Connection(self.input_nodes[i], gen)
        # Create connection handles for left and right motor neuron
        conn_l = nengo.Connection(gen.neurons, self.neuron_postLeft.neurons, transform=self.w_l[0][i])
        conn_r = nengo.Connection(gen.neurons, self.neuron_postRight.neurons, transform=self.w_r[0][i])
        # Create Probes
        self.generators_probe.append(nengo.Probe(gen.neurons))
        self.spike_generators.append(gen)
        self.connections_l.append(conn_l)
        self.connections_r.append(conn_r)
    def setWeights(self, weights_l, weights_r):
        factor = 250
        w_l = []
        for w in weights_l.reshape(weights_l.size):
            if w==0:    # for weight = 0 there is a problem in nengo when building the network
                w +=0.00001
            w_l.append((w*0.001)/factor) # because when building the network the weight is divided by dt (=0,001)
            # we need the factor to see, for which weights the network works best, since nengo doesn't work exactly like nest
        w_r = []
        for w in weights_r.reshape(weights_r.size):
            if w==0:
                w +=0.00001 # for weight = 0 there is a problem in nengo when building the network

            w_r.append((w*0.001)/factor)
        return np.array(w_l).reshape(1,32), np.array(w_r).reshape(1,32)
    def setInputData(self,arr):
        for i in range(len(self.input_data)):
            if arr[i] != 0:
                # tuning curves for each neuron is a tuple of two elements: one for output value corresponding with the firing rate
                if(len(self.tuningCurves)>0):
                    rate = arr[i] / p.max_spikes
                    rate = np.clip(rate, 0, 1) * p.max_poisson_freq
                    if rate >= 300:  # if bigger than max_rates
                        self.input_data[i] = 1  # for output = 1 the firing Rate is biggest, for output > 1 it wont get bigger
                    else:
                        self.input_data[i] = self.interpolation_function(rate)
                    continue
            self.input_data[i] = -1

