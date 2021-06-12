#include <string>
#include <vector>

using namespace std;

class Perceptron{

	public:
        float bias;
        float learning_rate;
        string* activation_function;
        string* loss_function;
        vector<float> weights;

        Perceptron(int input_size, float learning_rate, string* act_fun, string* loss_fun);
        void train(int iterations, vector<vector<float>> data, vector<float> target);
        vector<float> evaluate(vector<vector<float>> data, vector<float> target);
        float activationValue(vector<float> input);
        float activationFunction(float val) const;
        float derivativeActivationFunction(float val) const;
        float lossFunction(vector<float> real_out, vector<float> desired_out) const;
        float derivativeLossFunction(float real_out, float desired_out) const;
        void updateWeights(vector<float> gradients);
};
