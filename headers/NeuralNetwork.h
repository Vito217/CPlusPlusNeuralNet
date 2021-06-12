#include "Layer.h"
#include <iostream>     // std::cout
#include <tuple>

using namespace std;

class NeuralNetwork{

    private:
        vector<Layer> layers;
        vector<vector<vector<float>>> forwardProp(vector<vector<float>> data);
        vector<vector<vector<float>>> backwardProp(const vector<vector<float>>& data,
                                                   const vector<vector<float>>& desired_output,
                                                   vector<vector<vector<float>>> cache_list);
        vector<vector<vector<float>>> meanBackwardProp(vector<vector<float>> data,
                                                       vector<vector<vector<float>>> gradient_list,
                                                       vector<vector<vector<float>>> cache_list);
        void updateWeights(vector<vector<vector<float>>> mean_gradient_list);

    public:
        explicit NeuralNetwork(vector<Layer> ls);
        tuple<vector<float>, vector<float>> train(vector<vector<float>> data,
                                                  vector<vector<float>> desired_output,
                                                  int iterations);
        tuple<vector<vector<float>>, float, float> eval(vector<vector<float>> eval_data,
                                                        vector<vector<float>> eval_target);
};