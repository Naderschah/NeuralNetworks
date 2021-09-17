## NeuralNetworks

A repository containing a collection of NeuralNetworks implemented for StockPrice prediction

The final goal is developing an automated trading API using these networks



Developed Models:

MultiLayerPerceptron: Using non normalized input data from the SP500 stocks a maximum accuracy of ~67 % was achieved, however since this is correct identification Buy Hold and Sell signals it is not very representative. A more in depth analysis is found under model_analysis




Content:
- Analysis_Hyperparameter.py
    - To be implemented, suppose to run Hyperparameter optimization and save the best model for stock x if it does not exist yet
- OptimizeNN_keras.py
    - To be extended, collection of several architectures ready for Hyperparameter optimization
- Raw_Data.py
    - Function definitions to withdraw sets of data and format them for usage
- test.py
    - Scribbles
- Hyperparameter_optimization.ipynb
    - Further Scribbles