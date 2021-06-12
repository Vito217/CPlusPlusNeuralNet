#include "../headers/Perceptron.h"
#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>
#include <math.h>

Perceptron::Perceptron(int input_size, float learn_rate, string* act_fun, string* loss_fun)
{
	srand((unsigned int) time(nullptr));
	learning_rate = learn_rate;
	activation_function = act_fun;
	loss_function = loss_fun;
	weights.reserve(input_size);
    for(int i=0; i<input_size; i++)
		weights.push_back(-2 + 4*((float) rand()) / (float) RAND_MAX);
	this -> bias = -2 + 4*((float) rand()) /  (float) RAND_MAX;

}


float Perceptron::activationValue(vector<float> input)
{
    int input_size = input.size();
    int weights_size = weights.size();

	float res = bias;
	for(int i=0; i<input.size(); i++){
        float w = weights[i];
        float x = input[i];
	    res += w * x;
	}
	return res;
}

float Perceptron::activationFunction(float val) const
{
	if((*activation_function)=="step")
		return val <= 0 ? 0 : 1;
	if((*activation_function)=="sigmoid")
		return 1 / (1 + exp(-val));
	if((*activation_function)== "tanh")
		return (exp(val)-exp(-val)) / 
		       (exp(val)+exp(-val));
	return 0;
}

float Perceptron::derivativeActivationFunction(float val) const
{
	if((*activation_function)=="step")
		return 0;
	if((*activation_function)=="sigmoid")
		return val*(1-val);
	if((*activation_function)=="tanh")
		return 1 - pow((exp(val)-exp(-val))/ (exp(val) + exp(-val)), 2);
	return 0;
}

float Perceptron::lossFunction(vector<float> real_out, vector<float> desired_out) const
{
	if((*activation_function)=="mse")
	{
		float mse = 0;
		for(int i=0; i<real_out.size(); i++)
			mse += pow(real_out[i]-desired_out[i], 2);
		return mse/real_out.size();
	}
	if((*activation_function)=="cross")
	{
		float cross = 0;
		for(int i=0; i<real_out.size(); i++)
			cross -= desired_out[i] * (float) (log(real_out[i]) / log(2));
		return cross/real_out.size();
	}	
	return 0;
}

float Perceptron::derivativeLossFunction(float real_out, float desired_out) const
{
    if((*activation_function)=="mse")
        return (desired_out-real_out) * real_out * (1-real_out);
    if((*activation_function)=="cross")
        return (desired_out-real_out);
    return 0;
}

void Perceptron::train(int iterations, vector<vector<float>> data, vector<float> target)
{
    // Every iteration
    for(int it=0; it<iterations; it++)
    {
        // For each row in the dataset
        for (int i = 0; i < data.size(); i++)
        {
            // Compute output
            float real_out = activationFunction(activationValue(data[i]));
            // Get desired output
            float desired_out = target[i];
            // Update weights
            for(int k = 0; k<data[i].size(); k++)
                weights[k] = weights[k] + learning_rate * data[i][k] * derivativeLossFunction(real_out, desired_out);
            bias = bias + learning_rate * derivativeLossFunction(real_out, desired_out);
        }
    }
}

vector<float> Perceptron::evaluate(vector<vector<float>> data, vector<float> target)
{
    // Initialize counters
    float correct_answers = 0;
    float total_answers = 0;

    // Initialize prediction matrix
    vector<float> output;
    output.reserve(data.size());

    // For each row in data
    for(int i=0; i<data.size(); i++)
    {
        // Getting prediction
        float res = activationFunction(activationValue(data[i]));

        std::cout << "Desired = " + to_string(target[i]) << std::endl;
        std::cout << "Obtained = " + to_string(res) << std::endl;

        // Threshold is 0.5 by default
        if(round(res) == round(target[i]))
            correct_answers++;

        total_answers++;

        // Updating predictions matrix
        output[i] = res;
    }
    float acc = correct_answers / total_answers;
    std::cout << "Accuracy = " + to_string(acc);
    return output;
}

void Perceptron::updateWeights(vector<float> gradients)
{
    // Updating weights
    for(int i=0; i<gradients.size()-1; i++){
        weights[i] = weights[i] + learning_rate * gradients[i];
    }
    // Updating bias
    bias = bias + learning_rate * gradients[gradients.size() - 1];
}
