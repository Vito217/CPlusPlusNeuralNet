#include <string>
#include <vector>
#include "Perceptron.h"

using namespace std;

template<typename Base, typename T>
inline bool instanceof(const T*) {
    return is_base_of<Base, T>::value;
}

class BaseLayer{

};

class NullLayer : public BaseLayer{

};

class Layer : public BaseLayer{

    public:
        vector<Perceptron> neurons;
        Layer(int input_size, int n_neurons, float learning_rate, string* act_fun, string* loss_fun);
        vector<vector<float>> layerCache(vector<vector<float>> data);
        vector<vector<float>> layerGradients(const vector<vector<float>>& data, vector<vector<float>> layer_cache,
                                             vector<vector<float>> desired_output, Layer next_layer,
                                             vector<vector<float>> last_layer_gradient);
    vector<vector<float>> layerGradients(const vector<vector<float>>& data, vector<vector<float>> layer_cache,
                                         vector<vector<float>> desired_output, NullLayer next_layer,
                                         vector<vector<float>> last_layer_gradient);
        void updateWeights(vector<vector<float>> gradients);
        float lossFunction(vector<vector<float>> real_out, vector<vector<float>> desired_out);
        vector<float> evaluate(const vector<float>& data);

    private:
        string activation_function;
        string loss_function;
};
