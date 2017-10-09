#include <iostream>
#include <fstream>
#include <string>
#include <istream>
#include <vector>
#include <iterator>
#include <boost/algorithm/string.hpp>

using namespace std;

/*
* Helper functions to read training data
*/ 

struct train_type {
    vector<string> t_data;
    string t_class;
};

void print_train_data(vector<train_type> train) {
    for(int i=0; i<train.size(); i++) {
        for(int j=0; j<train[i].t_data.size(); j++) {
            cout << train[i].t_data[j] << " ";
        }
        cout << train[i].t_class << endl;
    }
}
 
vector<train_type> read_data(string input_file, string delim) {
    string line;
    ifstream ifile (input_file);
    vector<train_type> train;
    if(ifile.is_open()) {
        while(getline(ifile, line)) { 
            train_type tokens;
            boost::split(tokens.t_data, line, boost::is_any_of(delim));
            tokens.t_class = tokens.t_data[tokens.t_data.size()-1];
            tokens.t_data.pop_back();
            train.push_back(tokens);
        }
    }
    else {
        cout << "Unable to open the file\n";
    }
    return train;
}

vector<vector<double> > normalize(vector<vector<double> > input, int normal_val) {
    for(int i=0; i<input.size(); i++) {
        for(int j=0; j<input[i].size(); j++) {
            input[i][j] = input[i][j] * 1.0 / math.sqrt(normal_val);
        }
    }
    return input;
}

// Neural network

class neural_network {
    private:
        int hidden_nodes = 500;
        vector<vector<double> > w1, b1, w2, b2;
    
    public:
        void initialize();
        void train(vector<string> data, string cls);
        string test(vector<string> data);
};

void neural_network::initialize(int train_size, int output_size) {
    int n = train_size, s = hidden_nodes, p = output_size;
    w1.resize(n, vector<double>(s, rand()));
    w1 = normalize(w1, n);
    b1.resize(1, vector<double>(s, rand()));
    w2.resize(s, vector<double>(p, rand()));
    w2 = normalize(w2, s);
    b2.resize(1, vector<double<(p, rand()));
}

void neural_network::train(vector<string>



int main(int argc, char *argv[]) {
    string train_file = "../data/train.csv";
    vector<train_type> train = read_data(train_file, ",");
   
    // Training phase
    nn = neural_network();
    nn.initialize();
    for(int i=0; i<train.size(); i++) {
        vector<string> train_data = train[i].t_data;
        nn.train(train[i].t_data, train[i].t_class);
    }

    // Testing phase
    string test_file = "../data/test.csv";
    vector<train_type> test = read_data(test_file, ",");
    int cnt = 0;
    for(int i=0; i<test.size(); i++) {
        string predicted_class = nn.predict(test[i].t_data);
        if(predicted_class == test[i].t_class) { cnt = cnt + 1; }
    }

    printf("Accuracy percentage: %.2f\n", (cnt * 100.0 / test.size()));
    return 0;
}
