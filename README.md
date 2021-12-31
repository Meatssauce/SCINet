# SCINet:  Sample Convolution and Interaction Networks Implementation (Work in Progress)

This is an implementation of SCINet using tensorflow and a work in progress.
I want to explore the possibility of using SCINet to predict cryptocurrency prices and how they compare to traditional 
approaches such as an ARIMA.

SCINet is a novel architecture for time series forecasting proposed in this [paper](https://arxiv.org/pdf/2106.09305v1.pdf).
See original paper for link to datasets.

## Notes
 - See applications.testing.sinewave.py for usage examples
 - Obtained similar results on the ETD dataset (ETDataset-main/ETT-small/ETTh1.csv) used in the orignal paper but only with a batch size of 16 instead of 4. The cause of the discrepancy is unclear - pending investigation.
 - Scored poorly on crypto data (mse ~= 1.5, ase ~= 0.8 when data is relative difference). Learning curve suggests model is underfitting, which is expected as the data contains only a few basic features and has undergone minimal feature engineering. No hyperparamters tuning either. The score should serve as a baseline for future improvements.
