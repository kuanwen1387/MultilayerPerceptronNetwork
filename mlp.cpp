/********************************************************
*							*
*	CSCI426 Assignment 1				*
*	Student Name: Kuan Wen Ng			*
*	Subject Code: CSCI426				*
*	Student Number: 5078052				*
*	Email ID: kwn961				*
*	Filename: mlp.cpp (ass1)			*
*	Description: Multi Layer Perceptron Network	*
*							*
********************************************************/
//NOTE THAT THIS PROGRAM WONT WORK WITH NO HIDDEN LAYER
//AT LEAST ONE HIDDEN LAYER
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
using namespace std;

const int STRING_SIZE = 100;
const int MAX_NEURONS = 50;
const int MAX_PATTERN = 5000;

struct RandomIndex
{
	int index;
	bool result;

	//Assignment operator for RandomIndex
	const RandomIndex operator = (const RandomIndex &assign)
	{
		index = assign.index;
		result = assign.result;

		return *this;
	}
};

typedef vector<RandomIndex> Pair;

class MLP
{
private:
	int inputNum;
	int outputNum;
	int trainingNum;
	int testNum;
	int epoch;
	int hiddenLayerNum;
	int *hiddenLayerNeuron;
	int *patIndex;
	double learningRate;
	double momentum1;
	double momentum2;
	double objErr;
	int ordering;
	double **inputTrainingData;
	double **outputTrainingData;
	double **inputTestData;
	double **outputTestData;
	double ****weight;
	double **output;
	double **delta;
	Pair pair;

public:
	MLP();
	~MLP();
	bool readData();
	double **allocateMemory(int , int );
	void deleteMemory(double ** , int );
	void allocateWeights();
	void setOrdering();
	void permute();
	void randomSwap();
	void randomSelect();
	void normalize();
	void trainNet();
	void testNet();
};

MLP::MLP()
{
	inputNum = 0;
	outputNum = 0;
	trainingNum = 0;
	testNum = 0;
	epoch = 0;
	hiddenLayerNum = 0;
	hiddenLayerNeuron = NULL;
	patIndex = NULL;
	learningRate = 0;
	momentum1 = 0;
	momentum2 = 0;
	objErr = 0;
	ordering = 0;
	inputTrainingData = NULL;
	outputTrainingData = NULL;
	inputTestData = NULL;
	outputTestData = NULL;
	weight = NULL;
	output = NULL;
	delta = NULL;
}

MLP::~MLP()
{
	if (hiddenLayerNeuron != NULL)
		delete [] hiddenLayerNeuron;

	if (patIndex != NULL)
		delete [] patIndex;

	if (inputTrainingData != NULL)
	{
		deleteMemory(inputTrainingData, trainingNum);
		deleteMemory(outputTrainingData, trainingNum);
		deleteMemory(inputTestData, testNum);
		deleteMemory(outputTestData, testNum);
	}

	if (weight != NULL)
	{
		for (int copyIndex = 0; copyIndex < 3; copyIndex++)
		{
			for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
			{
				if (layerIndex == 0)
				{
					for (int neuronIndex = 0; neuronIndex < inputNum + 1; neuronIndex++)
						delete [] weight[copyIndex][layerIndex][neuronIndex];

					delete [] weight[copyIndex][layerIndex];
				}

				else if (layerIndex == hiddenLayerNum)
				{
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[hiddenLayerNum - 1] + 1; neuronIndex++)
						delete [] weight[copyIndex][layerIndex][neuronIndex];

					delete [] weight[copyIndex][layerIndex];
				}

				else
				{
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex - 1] + 1; neuronIndex++)
						delete [] weight[copyIndex][layerIndex][neuronIndex];

					delete [] weight[copyIndex][layerIndex];
				}
			}
			delete [] weight[copyIndex];
		}
		delete [] weight;
	}

	if (output != NULL)
	{
		for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
		{
			delete [] output[layerIndex];
			delete [] delta[layerIndex];
		}

		delete [] output;
		delete [] delta;
	}
}

//Function to read dataset
bool MLP::readData()
{
	char readString[STRING_SIZE], temp[STRING_SIZE];
	ifstream inputFile;
	int tempHiddenLayer = 0;
	bool success = false;

	//Prompt user for filename
	cout << "Please enter filename: ";
	cin.getline(readString, STRING_SIZE);

	//Open file
	inputFile.open(readString);

	if (inputFile.good())
		success = true;

	//Skip comments
	do
	{
		inputFile.getline(readString, STRING_SIZE, '\n');
	} while (readString[0] == ';');

	//Read header file
	sscanf(readString, "%s%d", temp, &inputNum);
	inputFile >> readString >> outputNum;
	inputFile >> readString >> trainingNum;
	inputFile >> readString >> testNum;
	inputFile >> readString >> epoch;
	inputFile >> readString >> hiddenLayerNum;

	//Read number of neurons for hidden layer
	hiddenLayerNeuron = new int[hiddenLayerNum];
	for (int index = 0; index < hiddenLayerNum; index++)
	{
		inputFile >> readString >> hiddenLayerNeuron[index];
	}

	inputFile >> readString >> learningRate;
	inputFile >> readString >> momentum1;
	inputFile >> readString >> momentum2;
	inputFile >> readString >> objErr;
	inputFile >> readString >> ordering;

	//Validate header
	if (inputNum < 1 || inputNum > MAX_NEURONS || outputNum < 1 || outputNum > MAX_NEURONS ||
		trainingNum < 1 || trainingNum > MAX_PATTERN || testNum < 1 || testNum > MAX_PATTERN ||
		epoch < 1 || epoch > 20000000 || learningRate < 0 || learningRate > 1 ||
		momentum1 < 0 || momentum1 > 10 || momentum2 < 0 || momentum2 > 10 || objErr < 0 || objErr > 10)
	{
		cout << "Invalid parameters in data file." << endl;
		return false;
	}

	for (int index = 0; index < hiddenLayerNum; index++)
	{
		if (hiddenLayerNeuron[index] < 0 || hiddenLayerNeuron[index] > MAX_PATTERN)
		{
			cout << "Invalid parameters in data file." << endl;
			return false;
		}
	}

	//Allocate dynamic memory for patterns
	inputTrainingData = allocateMemory(trainingNum, inputNum);
	outputTrainingData = allocateMemory(trainingNum, outputNum);
	inputTestData = allocateMemory(testNum, inputNum);
	outputTestData = allocateMemory(testNum, outputNum);

	for (int rowIndex = 0; rowIndex < trainingNum; rowIndex++)
	{
		for (int columnIndex = 0; columnIndex < inputNum; columnIndex++)
			inputFile >> inputTrainingData[rowIndex][columnIndex];

		for (int columnIndex = 0; columnIndex < outputNum; columnIndex++)
			inputFile >> outputTrainingData[rowIndex][columnIndex];
	}

	for (int rowIndex = 0; rowIndex < testNum; rowIndex++)
	{
		for (int columnIndex = 0; columnIndex < inputNum; columnIndex++)
			inputFile >> inputTestData[rowIndex][columnIndex];

		for (int columnIndex = 0; columnIndex < outputNum; columnIndex++)
			inputFile >> outputTestData[rowIndex][columnIndex];
	}

	//Close file
	inputFile.close();

	return success;
}

//Function to allocate memory for 2d array
double **MLP::allocateMemory(int row, int column)
{
	double **data = new double*[row];

	for (int index = 0; index < row; index++)
		data[index] = new double[column];

	return data;
}

//Function to delete 2d array
void MLP::deleteMemory(double **data, int row)
{
	for (int index = 0; index < row; index++)
		delete [] data[index];

	delete [] data;
}

//Fucntion to allocate and initialize weights
void MLP::allocateWeights()
{
	//Allocate dynamic memory for weight
	weight = new double***[3];

	//Allocate memory for 3 copies
	for (int copyIndex = 0; copyIndex < 3; copyIndex++)
	{
		//Weights for each layers
		weight[copyIndex] = new double**[hiddenLayerNum + 1];

		for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
		{
			//Weights from input to first hidden layer
			if (layerIndex == 0)
			{
				weight[copyIndex][layerIndex] = new double*[inputNum + 1]; //+1 element for bias

				for (int neuronIndex = 0; neuronIndex < inputNum + 1; neuronIndex++)	//+1 Bias
					weight[copyIndex][layerIndex][neuronIndex] = new double[hiddenLayerNeuron[0]];
			}

			//Weights from hidden layer to output layer
			else if (layerIndex == hiddenLayerNum)
			{
				weight[copyIndex][layerIndex] = new double*[hiddenLayerNeuron[hiddenLayerNum - 1] + 1];	//+1 Bias

				for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[hiddenLayerNum - 1] + 1; neuronIndex++)	//+1 Bias
					weight[copyIndex][layerIndex][neuronIndex] = new double[outputNum];
			}

			//Weights for every other layers
			else
			{
				weight[copyIndex][layerIndex] = new double*[hiddenLayerNeuron[layerIndex - 1] + 1];	//+1 Bias

				for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex - 1] + 1; neuronIndex++)	//+1 Bias
					weight[copyIndex][layerIndex][neuronIndex] = new double[hiddenLayerNeuron[layerIndex]];
			}
		}
	}

	//Initialize weight
	srand(time(0));

	//Weights for each layer
	for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
	{
		//Weights for input layer to hidden layer
		if (layerIndex == 0)
		{
			for (int neuronIndex = 0; neuronIndex < inputNum + 1; neuronIndex++)	//+1 Bias
			{
				for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[0]; weightIndex++)
					weight[0][layerIndex][neuronIndex][weightIndex] = weight[1][layerIndex][neuronIndex][weightIndex]
					= weight[2][layerIndex][neuronIndex][weightIndex] = double(rand())/RAND_MAX - 0.5;
			}
		}

		//Weights from hidden layer to output layer
		else if (layerIndex == hiddenLayerNum)
		{
			for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[hiddenLayerNum - 1] + 1; neuronIndex++)	//+1 Bias
			{
				for (int weightIndex = 0; weightIndex < outputNum; weightIndex++)
					weight[0][layerIndex][neuronIndex][weightIndex] = weight[1][layerIndex][neuronIndex][weightIndex]
					= weight[2][layerIndex][neuronIndex][weightIndex] = double(rand())/RAND_MAX - 0.5;
			}
		}

		//Weights for other layers
		else
		{
			for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex - 1] + 1; neuronIndex++)	//+1 Bias
			{
				for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex]; weightIndex++)
					weight[0][layerIndex][neuronIndex][weightIndex] = weight[1][layerIndex][neuronIndex][weightIndex]
					= weight[2][layerIndex][neuronIndex][weightIndex] = double(rand())/RAND_MAX - 0.5;
			}
		}
	}
}

//Ordering for each epoch
void MLP::setOrdering()
{
	switch (ordering)
	{
		case 1: permute();
			break;

		case 2: randomSwap();
			break;

		default: randomSelect();
	}
}

//Permutation for ordering
void MLP::permute()
{
	int min = 0, max = trainingNum - 1, tempIndex, tempNum;
	srand(time(0));

	for (int index = 0; index < trainingNum; index++)
	{
		tempIndex = (rand() % (max - min + 1)) + min;
		tempNum = patIndex[index];
		patIndex[index] = patIndex[tempIndex];
		patIndex[tempIndex] = tempNum;
	}
}

//Swap 2 patterns
void MLP::randomSwap()
{
	int min = 0, max = trainingNum - 1, tempIndex1, tempIndex2, tempNum;
	srand(time(0));

	tempIndex1 = (rand() % (max - min + 1)) + min;
	tempIndex2 = (rand() % (max - min + 1)) + min;

	while (tempIndex1 == tempIndex2)
		tempIndex2 = (rand() % (max - min + 1)) + min;

	tempNum = patIndex[tempIndex1];
	patIndex[tempIndex1] = patIndex[tempIndex2];
	patIndex[tempIndex2] = tempNum;
}

//Build order from correct pattern
void MLP::randomSelect()
{
	int setIndex = 0, tempIndex, tempNum, minValue = 0, maxValue, size = 3;
	bool select;

	while (pair.size() > 0)
	{
		select = false;

		for (int index = 0; index < size && !select; index++)
		{
			maxValue = pair.size() - 1;

			if (maxValue > 0)	
				tempIndex = (rand() % (maxValue - minValue + 1)) + minValue;
			else
				tempIndex = 0;
			if (pair.at(tempIndex).result)
			{
				patIndex[setIndex] = pair.at(tempIndex).index;
				pair.erase(pair.begin() + tempIndex);
				setIndex++;
				select = true;
			}


			if (index == 0 && !select)
				tempNum = tempIndex;

			if ((index == size -1) && !select)
			{
				patIndex[setIndex] = pair.at(tempIndex).index;
				pair.erase(pair.begin() + tempIndex);
				setIndex++;
				select = true;
			}
		}
	}
}

//Normalize input data
void MLP::normalize()
{
	double value[2][inputNum], tempNum;

	//Compute mean
	for (int columnIndex = 0; columnIndex < inputNum; columnIndex++)
	{
		tempNum = 0;

		for (int rowIndex = 0; rowIndex < trainingNum; rowIndex++)
			tempNum += inputTrainingData[rowIndex][columnIndex];

		value[0][columnIndex] = tempNum / trainingNum;
	}

	//Compute standard deviation
	for (int columnIndex = 0; columnIndex < inputNum; columnIndex++)
	{
		value[1][columnIndex] = 0;

		for (int rowIndex = 0; rowIndex < trainingNum; rowIndex++)
		{
			tempNum = inputTrainingData[rowIndex][columnIndex] - value[0][columnIndex];
			value[1][columnIndex] += pow(tempNum, 2.0);
		}

		value[1][columnIndex] = sqrt(value[1][columnIndex] / trainingNum);
	}

	//Normalize training data
	for (int columnIndex = 0; columnIndex < inputNum; columnIndex++)
	{
		for (int rowIndex = 0; rowIndex < trainingNum; rowIndex++)
		{
			inputTrainingData[rowIndex][columnIndex] = (inputTrainingData[rowIndex][columnIndex] - value[0][columnIndex]) / value[1][columnIndex];
		}
	}

	//Normalize test data
	for (int columnIndex = 0; columnIndex < inputNum; columnIndex++)
	{
		for (int rowIndex = 0; rowIndex < testNum; rowIndex++)
		{
			inputTestData[rowIndex][columnIndex] = (inputTestData[rowIndex][columnIndex] - value[0][columnIndex]) / value[1][columnIndex];
		}
	}
}

//Training function for MLP
void MLP::trainNet()
{
	double patErr, minErr, avgErr, maxErr, pctCor, input, tempWeight, error;
	int epochCounter = 0, numErr;
	RandomIndex randomIndex;

	//Allocate memory for patIndex
	patIndex = new int[trainingNum];
	for (int index = 0; index < trainingNum; index++)
		patIndex[index] = index;

	//Allocate memory for output and delta
	output = new double*[hiddenLayerNum + 1];
	delta = new double*[hiddenLayerNum + 1];

	for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
	{
		if (layerIndex == hiddenLayerNum)
		{
			output[layerIndex] = new double[outputNum];
			delta[layerIndex] = new double[outputNum];
		}

		else
		{
			output[layerIndex] = new double[hiddenLayerNeuron[layerIndex]];
			delta[layerIndex] = new double[hiddenLayerNeuron[layerIndex]];
		}
	}

	//Random ordering
	permute();

	//Normalize input
	normalize();

	//Output network architecture
	cout << "Network Architecture"
	<< "\nInput: " << inputNum << endl;
	for (int index = 0; index < hiddenLayerNum; index++)
		cout << "Hidden Layer " << index + 1 << ": " << hiddenLayerNeuron[index] << endl;

	cout << "Output: " << outputNum
	<< "\nLearning Rate: " << learningRate
	<< "\nMomentum 1: " << momentum1
	<< "\nMomentum 2: " << momentum2
	<< "\nObject Error: " << objErr
	<< "\nOrdering: " << ordering << endl;

	do
	{
		minErr = 3.4e38;
		avgErr = 0;
		maxErr = -3.4e38;
		numErr = 0;

		//Set ordering
		if (ordering > 0 && epochCounter > 0)
			setOrdering();

		//Forward phase and each epoch starts here
		for (int patternIndex = 0; patternIndex < trainingNum; patternIndex++)
		{
			//Compute output for each layer
			for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
			{
				//Output from input layer to hidden layer
				if (layerIndex == 0)
				{
					//Compute output for for each neuron in layer
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[0]; neuronIndex++)
					{
						input = 0;

						//Compute sum of weight * output of each neuron from input layer
						for (int weightIndex = 0; weightIndex < inputNum + 1; weightIndex++)	//+1 Bias
						{
							//Weight of bias
							if (weightIndex == inputNum)
								input += weight[0][layerIndex][weightIndex][neuronIndex];

							else
								input += weight[0][layerIndex][weightIndex][neuronIndex] * inputTrainingData[patIndex[patternIndex]][weightIndex];
						}

						//Output of neuron in hidden layer
						output[layerIndex][neuronIndex] = 1.0/(1.0+exp(double(-input)));
					}
				}

				//Output from hidden layer to output layer
				else if (layerIndex == hiddenLayerNum)
				{
					for (int neuronIndex = 0; neuronIndex < outputNum; neuronIndex++)
					{
						input = 0;

						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[hiddenLayerNum - 1] + 1; weightIndex++)
						{
							//Weight of bias
							if (weightIndex == hiddenLayerNeuron[hiddenLayerNum - 1])
								input += weight[0][layerIndex][weightIndex][neuronIndex];

							else
								input += weight[0][layerIndex][weightIndex][neuronIndex] * output[layerIndex - 1][weightIndex];
						}

						output[layerIndex][neuronIndex] = 1.0/(1.0+exp(double(-input)));
					}
				}

				//Output from hidden layer to the next hidden layer
				else
				{
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex]; neuronIndex++)
					{
						input = 0;

						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex - 1] + 1; weightIndex++)
						{
							//Weight of bias
							if (weightIndex == hiddenLayerNeuron[layerIndex - 1])
								input += weight[0][layerIndex][weightIndex][neuronIndex];

							else
								input += weight[0][layerIndex][weightIndex][neuronIndex] * output[layerIndex - 1][weightIndex];
						}

						output[layerIndex][neuronIndex] = 1.0/(1.0+exp(double(-input)));
					}
				}
			}

			//Compute error for pattern
			patErr = 0;

			for (int index = 0; index < outputNum; index++)
			{
				patErr = fabs(output[hiddenLayerNum][index] - outputTrainingData[patIndex[patternIndex]][index]);
				numErr += ((output[hiddenLayerNum][index] < 0.5 && outputTrainingData[patIndex[patternIndex]][index] >= 0.5) ||
					(output[hiddenLayerNum][index] >= 0.5 && outputTrainingData[patIndex[patternIndex]][index] < 0.5));

				//Store result for random selection
				if (ordering == 3)
				{
					randomIndex.index = patIndex[index];
					randomIndex.result = ((output[hiddenLayerNum][index] < 0.5 && outputTrainingData[patIndex[patternIndex]][index] >= 0.5) ||
						(output[hiddenLayerNum][index] >= 0.5 && outputTrainingData[patIndex[patternIndex]][index] < 0.5));
				}
			}

			if(patErr < minErr)
				minErr = patErr;

			if(patErr > maxErr)
				maxErr = patErr;

			avgErr += patErr;

			//Back propagation and change weights for each layer from output layer to input layer
			for (int layerIndex = hiddenLayerNum; layerIndex >= 0; layerIndex--)
			{
				//Adjust weights for output layer
				if (layerIndex == hiddenLayerNum)
				{
					//Compute delta for each output neuron
					for (int neuronIndex = 0; neuronIndex < outputNum; neuronIndex++)
					{
						delta[hiddenLayerNum][neuronIndex] = (outputTrainingData[patIndex[patternIndex]][neuronIndex]
						- output[hiddenLayerNum][neuronIndex]) * output[hiddenLayerNum][neuronIndex] * (1.0 - output[hiddenLayerNum][neuronIndex]);

						//Adjust each weights connected to the neuron
						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex - 1] + 1; weightIndex++) //+1 Bias
						{
							tempWeight = weight[0][layerIndex][weightIndex][neuronIndex];

							//Adjust the weight of bias
							if (weightIndex == hiddenLayerNeuron[layerIndex - 1])
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							else
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * output[layerIndex - 1][weightIndex]
														* delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							//Store previous weights in 2nd and 3rd copy for momentum
							weight[2][layerIndex][weightIndex][neuronIndex] = weight[1][layerIndex][weightIndex][neuronIndex];
							weight[1][layerIndex][weightIndex][neuronIndex] = tempWeight;
						}
					}
				}

				//Change weights to hidden layer before output layer
				else if (layerIndex == hiddenLayerNum - 1)
				{
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex]; neuronIndex++)
					{
						error = 0;

						//Compute sum of delta * weight of each output neuron
						for (int weightIndex = 0; weightIndex < outputNum; weightIndex++)
							error += delta[layerIndex + 1][weightIndex] * weight[1][layerIndex + 1][weightIndex][neuronIndex];

						//Delta of neuron in hidden layer
						delta[layerIndex][neuronIndex] = error * output[layerIndex][neuronIndex] * (1.0 - output[layerIndex][neuronIndex]);

						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex - 1] + 1; weightIndex++)
						{
							tempWeight = weight[0][layerIndex][weightIndex][neuronIndex];

							if (weightIndex == hiddenLayerNeuron[layerIndex - 1])
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							else
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * output[layerIndex - 1][weightIndex]
														* delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							weight[2][layerIndex][weightIndex][neuronIndex] = weight[1][layerIndex][weightIndex][neuronIndex];
							weight[1][layerIndex][weightIndex][neuronIndex] = tempWeight;
						}
					}
				}

				//Adjust weights from input layer to hidden layer
				else if (layerIndex == 0)
				{
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex]; neuronIndex++)
					{
						error = 0;

						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex + 1]; weightIndex++)
							error += delta[layerIndex + 1][weightIndex] * weight[1][layerIndex + 1][weightIndex][neuronIndex];

						delta[layerIndex][neuronIndex] = error * output[layerIndex][neuronIndex] * (1 - output[layerIndex][neuronIndex]);

						for (int weightIndex = 0; weightIndex < inputNum + 1; weightIndex++)
						{
							tempWeight = weight[0][layerIndex][weightIndex][neuronIndex];

							if (weightIndex == inputNum)
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							else
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate
														* inputTrainingData[patIndex[patternIndex]][weightIndex]
														* delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							weight[2][layerIndex][weightIndex][neuronIndex] = weight[1][layerIndex][weightIndex][neuronIndex];
							weight[1][layerIndex][weightIndex][neuronIndex] = tempWeight;
						}
					}
				}

				//Adjust weights for other layers
				else
				{
					for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex]; neuronIndex++)
					{
						error = 0;

						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex + 1]; weightIndex++)
							error += delta[layerIndex + 1][weightIndex] * weight[1][layerIndex + 1][weightIndex][neuronIndex];

						delta[layerIndex][neuronIndex] = error * output[layerIndex][neuronIndex] * (1 - output[layerIndex][neuronIndex]);

						for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex - 1] + 1; weightIndex++)
						{
							tempWeight = weight[0][layerIndex][weightIndex][neuronIndex];

							if (weightIndex == hiddenLayerNeuron[layerIndex - 1])
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							else
							{
								weight[0][layerIndex][weightIndex][neuronIndex] += learningRate * output[layerIndex - 1][weightIndex]
														* delta[hiddenLayerNum][neuronIndex]
														+ momentum1 * (weight[0][layerIndex][weightIndex][neuronIndex]
														- weight[1][layerIndex][weightIndex][neuronIndex])
														+ momentum2 * (weight[1][layerIndex][weightIndex][neuronIndex]
														- weight[2][layerIndex][weightIndex][neuronIndex]);
							}

							weight[2][layerIndex][weightIndex][neuronIndex] = weight[1][layerIndex][weightIndex][neuronIndex];
							weight[1][layerIndex][weightIndex][neuronIndex] = weight[0][layerIndex][weightIndex][neuronIndex];;
						}
					}
				}
			}//end for one epoch
		}

		epochCounter++;
		avgErr /= trainingNum;
		pctCor = (trainingNum - numErr) / double(trainingNum * outputNum) * 100.0;
		cout.setf(ios::fixed|ios::showpoint);

		if (epochCounter == 1)
		{
			cout << setw(6) << "#" << "  " << setw(12) << "MinErr" << setw(12) << "AvgErr" << setw(12) << "MaxErr" << setw(12) << "Correct %" << endl;
		}

    		cout << setprecision(6) << setw(6) << epochCounter << ": " << setw(12) << minErr << setw(12) << avgErr
		<< setw(12) << maxErr << setw(12) << pctCor << endl;
	} while ((epochCounter < epoch) && (avgErr > objErr));	//Stop training when max epochs or average error small
}

void MLP::testNet()
{
	int numErr = 0;
	double patErr, minErr = 3.4e38, avgErr = 0, maxErr = -3.4e38, pctCor = 0, input;

	for (int patternIndex = 0; patternIndex < testNum; patternIndex++)
	{
		//Compute output for each layer
		for (int layerIndex = 0; layerIndex < hiddenLayerNum + 1; layerIndex++)
		{
			//Output from input layer to hidden layer
			if (layerIndex == 0)
			{
				//Compute output for for each neuron in layer
				for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[0]; neuronIndex++)
				{
					input = 0;

					//Compute sum of weight * output of each neuron from input layer
					for (int weightIndex = 0; weightIndex < inputNum + 1; weightIndex++)	//+1 Bias
					{
						//Weight of bias
						if (weightIndex == inputNum)
							input += weight[0][layerIndex][weightIndex][neuronIndex];

						else
							input += weight[0][layerIndex][weightIndex][neuronIndex] * inputTestData[patternIndex][weightIndex];
					}

					//Output of neuron in hidden layer
					output[layerIndex][neuronIndex] = 1.0/(1.0+exp(double(-input)));
				}
			}

			//Output from hidden layer to output layer
			else if (layerIndex == hiddenLayerNum)
			{
				for (int neuronIndex = 0; neuronIndex < outputNum; neuronIndex++)
				{
					input = 0;

					for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[hiddenLayerNum - 1] + 1; weightIndex++)
					{
						//Weight of bias
						if (weightIndex == hiddenLayerNeuron[hiddenLayerNum - 1])
							input += weight[0][layerIndex][weightIndex][neuronIndex];

						else
							input += weight[0][layerIndex][weightIndex][neuronIndex] * output[layerIndex - 1][weightIndex];
					}

					output[layerIndex][neuronIndex] = 1.0/(1.0+exp(double(-input)));
				}
			}

			//Output from hidden layer to the next hidden layer
			else
			{
				for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuron[layerIndex]; neuronIndex++)
				{
					input = 0;

					for (int weightIndex = 0; weightIndex < hiddenLayerNeuron[layerIndex - 1] + 1; weightIndex++)
					{
						//Weight of bias
						if (weightIndex == hiddenLayerNeuron[layerIndex - 1])
							input += weight[0][layerIndex][weightIndex][neuronIndex];

						else
							input += weight[0][layerIndex][weightIndex][neuronIndex] * output[layerIndex - 1][weightIndex];
					}

					output[layerIndex][neuronIndex] = 1.0/(1.0+exp(double(-input)));
				}
			}
		}

		//Compute error for pattern
		patErr = 0;

		for (int index = 0; index < outputNum; index++)
		{
			patErr = fabs(output[hiddenLayerNum][index] - outputTestData[patternIndex][index]);
			numErr += ((output[hiddenLayerNum][index] < 0.5 && outputTestData[patternIndex][index] >= 0.5) ||
				(output[hiddenLayerNum][index] >= 0.5 && outputTestData[patternIndex][index] < 0.5));
		}

		if(patErr < minErr)
			minErr = patErr;

		if(patErr > maxErr)
			maxErr = patErr;

		avgErr += patErr;
	}
	avgErr /= testNum;
	pctCor = (testNum - numErr) / double(testNum * outputNum) * 100.0;
	cout.setf(ios::fixed|ios::showpoint);
	cout << "\nTesting MLP:\n" << setw(6) << " " << "  " << setw(12) << "MinErr" << setw(12) << "AvgErr" << setw(12) << "MaxErr" << setw(12) << "Correct %" << endl;
    	cout << setprecision(6) << setw(6) << "" << "  " << setw(12) << minErr << setw(12) << avgErr
	<< setw(12) << maxErr << setw(12) << pctCor << endl;
}

int main()
{
	MLP test;

	if (test.readData())
	{
		test.allocateWeights();
		test.trainNet();
		test.testNet();
	}


	return 0;

}
