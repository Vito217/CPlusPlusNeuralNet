//
// Created by vitos on 23-05-2021.
//

#include <headers/DataUtils.h>

vector<vector<float>> transpose(vector<vector<float>> matrix)
{
    vector<vector<float>> aux;
    aux.reserve(matrix[0].size());

    for(int j=0; j<matrix[0].size(); j++){

        vector<float> aux_row;
        aux_row.reserve(matrix.size());

        for(int k=0; k<matrix.size(); k++){
            //aux[j][k] = matrix[k][j];
            aux_row.push_back(matrix[k][j]);
        }

        aux.push_back(aux_row);
    }
    return aux;
}

