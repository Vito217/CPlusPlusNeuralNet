//
// Created by vitos on 23-05-2021.
//

#include <math.h>
#include <headers/NeuralNetwork.h>
#include <headers/DataUtils.h>

using namespace std;

NeuralNetwork::NeuralNetwork(vector<Layer> ls) {
    layers = move(ls);
}

vector<vector<vector<float>>> NeuralNetwork::forwardProp(vector<vector<float>> data) {

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

    // Initialize cache list
    vector<vector<vector<float>>> cache_list;
    int layers_size = layers.size();
    cache_list.reserve(layers_size);

    vector<vector<float>> first_cache = layers[0].layerCache(move(data));
    cache_list.push_back(first_cache);

    // For each layer remaining in the network
    for(int i=1; i<layers.size(); i++){

        // We get the layer cache
        vector<vector<float>> input = cache_list[i-1];
        vector<vector<float>> aux_data;
        int column_size = (input[0]).size();
        aux_data.reserve(column_size);

        for(int j=0; j<input[0].size(); j++)
        {
            vector<float> aux_data_row;
            aux_data_row.reserve(input.size());

            for(int k=0; k<input.size(); k++)
            {
                aux_data_row.push_back(input[k][j]);
            }

            aux_data.push_back(aux_data_row);
        }

        cache_list.push_back(layers[i].layerCache(aux_data));
    }

    return cache_list;
}

vector<vector<vector<float>>>
NeuralNetwork::backwardProp(const vector<vector<float>>& data, const vector<vector<float>>& desired_output,
                            vector<vector<vector<float>>> cache_list) {

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

    vector<vector<vector<float>>> gradient_list;

    NullLayer nullLayer;
    vector<vector<float>> emptyGradients;
    emptyGradients.reserve(0);

    // First, we get gradient for last layer
    vector<vector<float>> last_layer_gradient =
           layers[layers.size()-1].layerGradients(
                   data,
                   cache_list[layers.size()-1],
                   desired_output,
                   nullLayer,
                   emptyGradients);

    gradient_list.push_back(last_layer_gradient);

    // For the remaining layers, from finish to start
    for(int i= (int) layers.size()-2; i>=0; i--){

        // Get layer gradient using next layer's weights and gradients
        vector<vector<float>> newLastGradient =
                layers[i].layerGradients(
                        data,
                        cache_list[i],
                        desired_output,
                        layers[i+1],
                        last_layer_gradient);

        int s = newLastGradient.size();

        gradient_list.insert(gradient_list.begin(), newLastGradient);
        last_layer_gradient = newLastGradient;
    }

    return meanBackwardProp(data, gradient_list, cache_list);
}

vector<vector<vector<float>>>
NeuralNetwork::meanBackwardProp(
        vector<vector<float>> data,
        vector<vector<vector<float>>> gradient_list,
        vector<vector<vector<float>>> cache_list) {

    // Final Gradient List for a single layer:
    //
    //             weights + bias
    //         _ _ _ _ _ _ _ _ _ _ _
    // n    i1|
    // e    i2|
    // u     .|
    // r     .|
    // o     .|
    // n     .|
    // s    in|_  _  _  _ _ _ _ _ _
    //         j1 j2 j3 . . . . . jd

    vector<vector<vector<float>>> mean_gradient_list;

    // For each layer
    for(int l=0; l<layers.size(); l++){

        Layer layer = layers[l];
        vector<vector<float>> layer_gradient = gradient_list[l];
        vector<vector<float>> mean_layer_gradient;
        mean_layer_gradient.reserve(layer.neurons.size());

        // For each neuron in the layer
        for(int i=0; i<layer.neurons.size(); i++){
            // For each weight of the neuron

            vector<float> aux_row;

            for(int j=0; j<layer.neurons[i].weights.size(); j++){
                // For each input that enters the layer

                float sum = 0;

                if(l==0){
                    // First layer recieves data itself
                    /**
                    for(int k=0; k<data.size(); k++)
                    {
                        float input_val = data[k][j];
                        float grad = layer_gradient[i][k];
                        mean_layer_gradient[i][j] += input_val * grad / data.size();
                    }
                     **/

                    for(int k=0; k<data.size(); k++)
                        sum += data[k][j] * layer_gradient[i][k]
                                / data.size();

                }
                else{
                    // Inner and output layer recieve cache as input
                    /**
                    vector<vector<float>> previous_layer_cache = cache_list[l-1];
                    for(int k=0; k<previous_layer_cache[0].size(); k++){
                        mean_layer_gradient[i][j] +=
                                previous_layer_cache[j][k]*layer_gradient[i][k]/data.size();
                    }
                    **/
                    for(int k=0; k<cache_list[l-1][0].size(); k++)
                        sum += cache_list[l-1][j][k] * layer_gradient[i][k]
                                    / data.size();
                }
                aux_row.push_back(sum);
            }
            // Finally, we get the mean gradient for the bias
            float sum = 0;
            if(l==0){
                /**
                for(int k=0; k<data.size(); k++){
                    mean_layer_gradient[i][layer.neurons[0].weights.size()] +=
                            layer_gradient[i][k]/data.size();
                }
                **/
                for(int k=0; k<data.size(); k++)
                    sum += layer_gradient[i][k]/data.size();
            }
            else{
                /**
                vector<vector<float>> previous_layer_cache = cache_list[l];
                for(int k=0; k<previous_layer_cache[0].size(); k++){
                    mean_layer_gradient[i][layer.neurons[0].weights.size()] +=
                            layer_gradient[i][k]/data.size();
                }
                **/
                for(int k=0; k<cache_list[l][0].size(); k++)
                    sum += layer_gradient[i][k]/data.size();
            }
            aux_row.push_back(sum);
            mean_layer_gradient.push_back(aux_row);
        }
        mean_gradient_list.push_back(mean_layer_gradient);
    }
    return mean_gradient_list;
}

void NeuralNetwork::updateWeights(vector<vector<vector<float>>> mean_gradient_list) {
    // Updating weights
    for(int i=0; i<layers.size(); i++){
        layers[i].updateWeights(mean_gradient_list[i]);
    }
}

tuple<vector<float>, vector<float>>
NeuralNetwork::train(vector<vector<float>> data, vector<vector<float>> desired_output, int iterations) {

    // For each iteration
    vector<float> loss;
    vector<float> success;
    loss.reserve(iterations);
    success.reserve(iterations);

    for(int it=0; it<iterations; it++){
        std::cout << "Step = " + to_string(it+1) << '\n';
        // Forward propagation
        vector<vector<vector<float>>> cache_list = forwardProp(data);
        // Backward propagation
        vector<vector<vector<float>>> gradient_list = backwardProp(data, desired_output, cache_list);
        // Update weights
        updateWeights(gradient_list);
        // Get loss, and right answers

        int s = data.size();

        tuple<vector<vector<float>>, float, float> out_ans_loss = eval(data, desired_output);

        success[it] = get<1>(out_ans_loss);
        loss[it] = get<2>(out_ans_loss);
    }

    tuple<vector<float>, vector<float>> t(loss, success);
    return t;
}

tuple<vector<vector<float>>, float, float> NeuralNetwork::eval(
        vector<vector<float>> eval_data,
        vector<vector<float>> eval_target) {

    // Initialize counters
    float correct_answers = 0;
    float total_answers = 0;

    // Initialize prediction matrix
    vector<vector<float>>  outputs;
    outputs.reserve(eval_target.size());

    // For each row in data
    for(int i=0; i<eval_data.size(); i++){

        // Getting prediction
        vector<float> output = eval_data[i];
        for(Layer layer: layers){
            output = layer.evaluate(output);
        }
        outputs.push_back(output);

        // Threshold is 0.5 by default
        for(int j=0; j<output.size(); j++){
            if(round(output[j]) == round(eval_target[i][j])){
                correct_answers += 1.0f/output.size();
            }
        }
        total_answers++;
    }

    float acc = correct_answers / total_answers;
    float loss = layers[layers.size()-1].lossFunction(transpose(outputs), transpose(eval_target));

    std::cout << "Correct Answers = " + to_string(round(correct_answers)) << '\n';
    std::cout << "Loss = "+ to_string(loss) << '\n';
    std::cout << "Accuracy = "+ to_string(acc) << '\n';

    tuple<vector<vector<float>>, float, float> t(outputs, correct_answers, loss);
    return t;
}
