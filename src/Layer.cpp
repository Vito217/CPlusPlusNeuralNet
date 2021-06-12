#include <headers/Layer.h>

Layer::Layer(int input_size, int n_neurons, float learning_rate, string* act_fun, string* loss_fun)
{
    neurons.reserve(n_neurons);
    for(int i=0; i<n_neurons; i++){
        auto p = new Perceptron(input_size, learning_rate, act_fun, loss_fun);
        neurons.push_back(*p);
    }
    activation_function = *act_fun;
    loss_function = *loss_fun;
}

vector<vector<float>> Layer::layerCache(vector<vector<float>> data)
{
    // Cache List for a single layer:
    //
    //    outputs (columns for each data row)
    //         _ _ _ _ _ _ _ _ _ _ _
    // n    i1|
    // e    i2|
    // u     .|
    // r     .|
    // o     .|
    // n     .|
    // s    in|_  _  _  _ _ _ _ _ _
    //         j1 j2 j3 . . . . . jm

    vector<vector<float>> layer_cache;
    layer_cache.reserve(neurons.size());

    int neurons_size = neurons.size();

    // For each neuron in the layer
    for(int i=0; i<neurons.size(); i++)
    {
        vector<float> caches;
        caches.reserve(data.size());

        // For each row in the dataset
        for(int j=0; j<data.size(); j++)
        {
            // We get the output from the neuron
            float activation_value = neurons[i].activationValue(data[j]);
            float cache =  neurons[i].activationFunction(activation_value);
            caches.push_back(cache);
        }
        layer_cache.push_back(caches);
    }
    return layer_cache;
}

vector<vector<float>> Layer::layerGradients(const vector<vector<float>>& data, vector<vector<float>> layer_cache,
                                            vector<vector<float>> desired_output, NullLayer next_layer,
                                            vector<vector<float>> last_layer_gradient)
{
    // Gradient List for a single layer:
    //
    //    outputs (columns for each data row)
    //         _ _ _ _ _ _ _ _ _ _ _
    // n    i1|
    // e    i2|
    // u     .|
    // r     .|
    // o     .|
    // n     .|
    // s    in|_  _  _  _ _ _ _ _ _
    //         j1 j2 j3 . . . . . jd

    vector<vector<float>> layer_gradient;
    layer_gradient.reserve(neurons.size());

    // If this is the last layer, we obtain gradient normally
    // i.e., err_fun_derivative * act_fun_derivative

    // For each neuron in the layer
    for(int i=0; i<neurons.size(); i++)
    {
        vector<float> gradients;
        gradients.reserve(data.size());

        // For each input that enters the neuron
        for(int j=0; j<data.size();j++)
        {
            // We get the delta
            gradients.push_back(
                    neurons[i].derivativeLossFunction(
                            layer_cache[i][j], desired_output[j][i]));
        }
        layer_gradient.push_back(gradients);
    }
    return layer_gradient;
}

vector<vector<float>> Layer::layerGradients(const vector<vector<float>>& data, vector<vector<float>> layer_cache,
                                            vector<vector<float>> desired_output, Layer next_layer,
                                            vector<vector<float>> last_layer_gradient)
{
    // Gradient List for a single layer:
    //
    //    outputs (columns for each data row)
    //         _ _ _ _ _ _ _ _ _ _ _
    // n    i1|
    // e    i2|
    // u     .|
    // r     .|
    // o     .|
    // n     .|
    // s    in|_  _  _  _ _ _ _ _ _
    //         j1 j2 j3 . . . . . jd

    vector<vector<float>> layer_gradient;
    layer_gradient.reserve(neurons.size());

    // For each neuron
    for(int i=0; i<neurons.size(); i++)
    {
        vector<float> gradients;
        gradients.reserve(data.size());

        // For each input that enters the neuron
        for(int j=0; j<data.size();j++){

            // First, we get the sum of delta * weights for the next layer
            float sum = 0;
            for(int k=0; k < next_layer.neurons.size(); k++)
            {
                // last_layer_gradient: from next layer, neuron k, input j
                // next_layer.neurons[k].weights[i]: we use the weight i that
                // receives the input from neuron i.
                float llg_value = last_layer_gradient[k][j];
                sum += llg_value * next_layer.neurons[k].weights[i];
            }

            // Then, we get the gradient
            gradients.push_back(
                    sum * neurons[i].derivativeActivationFunction(
                            layer_cache[i][j]));
        }
        layer_gradient.push_back(gradients);
    }

    int s = layer_gradient.size();

    return layer_gradient;
}

void Layer::updateWeights(vector<vector<float>> gradients)
{
    for(int i=0; i<gradients.size(); i++){
        neurons[i].updateWeights(gradients[i]);
    }
}

float Layer::lossFunction(vector<vector<float>> real_out, vector<vector<float>> desired_out)
{
    float loss = 0;
    for(int i=0; i<real_out.size(); i++){
        float neuron_mean_loss = neurons[i].lossFunction(real_out[i], desired_out[i]);
        if (loss_function == "mse")
            loss += neuron_mean_loss/real_out.size();
        else if(loss_function == "cross")
            loss += neuron_mean_loss;
    }
    return loss;
}

vector<float> Layer::evaluate(const vector<float>& data)
{
    vector<float> output;
    output.reserve(neurons.size());
    for(int i=0; i<neurons.size(); i++){
        output[i] = neurons[i].activationFunction(neurons[i].activationValue(data));
    }
    return output;
}

