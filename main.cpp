#include <iostream>
#include <headers/NeuralNetwork.h>

using namespace std;

int main()
{
    vector<float> d1{0.0, 0.0};
    vector<float> d2{0.0, 1.0};
    vector<float> d3{1.0, 0.0};
    vector<float> d4{1.0, 1.0};

    vector<vector<float>> data{d1,d2,d3, d4};

    vector<float> t1{0.0};
    vector<float> t2{0.0};
    vector<float> t3{0.0};
    vector<float> t4{1.0};

    vector<vector<float>> target{t1,t2,t3,t4};

    string sigmoid = "sigmoid";
    string cross = "mse";

    Layer l1(
            4,
            2,
            0.1f,
            &sigmoid,
            &cross
            );

    Layer l2(
            2,
            4,
            0.1f,
            &sigmoid,
            &cross
            );

    vector<Layer> layers{l1, l2};


    NeuralNetwork net(layers);

    net.train(data, target, 20);

}
