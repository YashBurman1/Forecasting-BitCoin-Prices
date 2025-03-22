# Forecasting-BitCoin-Prices

## 1 Importing and Exploring the Data Set
### 1.1 Data Cleaning
For this project I have used uhistorical prices on bitcoin available on kaggle.
The link to the dataset is https://www.kaggle.com/datasets/nisargchodavadiya/bitcoin-time-series-with-different-time-intervals. The first few columns of the data set are attached bwlow:
	Open	High	Low	Close	Adj Close	Volume
Date						
2020-01-01 00:00:00+00:00	7194.892090	7254.330566	7174.944336	7200.174316	7200.174316	1.856566e+10
2020-01-02 00:00:00+00:00	7202.551270	7212.155273	6935.270020	6985.470215	6985.470215	2.080208e+10
2020-01-03 00:00:00+00:00	6984.428711	7413.715332	6914.996094	7344.884277	7344.884277	2.811148e+10
2020-01-04 00:00:00+00:00	7345.375488	7427.385742	7309.514160	7410.656738	7410.656738	1.844427e+10
2020-01-05 00:00:00+00:00	7410.451660	7544.497070	7400.535645	7411.317383	7411.317383	1.972507e+10

In our analysis we will only be interested in the "Adj Close" column and build our model to predict the same. 

Since we'll primariliy be using the ARIMA function from the pmdarima library which directly supports datetime indices, we set the Date column as the index and drop the other column that we wont be using.
After doing so our data set looks like this:

	                        Adj Close
Date	
2020-01-01 00:00:00+00:00	7200.174316
2020-01-02 00:00:00+00:00	6985.470215
2020-01-03 00:00:00+00:00	7344.884277
2020-01-04 00:00:00+00:00	7410.656738
2020-01-05 00:00:00+00:00	7411.317383

### 1.2 Data Preprocessing 

To apply time series models we must first check if out series is stationary or not, to do so we use the the "Dickey Fuller Test" and use the implemtation of the same provided in the statsmodels libraby.
The resukts of the test were:
Augmented Dickey-Fuller Test Results:

ADF Statistic: -0.317683
P-Value: 0.922962
Used Lags: 1
Number of Observations: 646
Critical Values:
   1%: -3.440513
   5%: -2.866024
   10%: -2.569158

Conclusion: The time series is non-stationary (fail to reject H0).

Therfore, when we use XGBoost (we assumes stationarity) we will have to use differencing to convert the series into a stationary series.

## 2. Data Visualisation

We perform the very basic and elementary Data Visualisations, visualising how the series behaves and its components.
First we plot a BoxPlot of the the "Adj Close" aggregated over years. The resulting boxplot is :
//Insert Boxplot_Adj_Close here //
this indicated that in 2020, bitcoin prices had relatively less fluctuations and had multiple instances where the price reached abnormally high levels (as indicated by a large number of outliers) which could have signalled the upward breakout that was ultimately seen in 2021 where BitCoin Prices broke out out the $10,000 levels and grew to an average of $45,000 levels.

We plot a simular boxplot but here we aggregate the series over month for each year. 
// Insert Image//
Thse box plots reveal that the rally in the BitCoin prices started around october 2020 and continued till April 2021. 

Next we use seasonal_decompose function in the statsmodel library to decompose out series in the following components:
  1. Long Term Trend
  2. Seasonal Fluctuation
  3. Rondom movements
Assuming ann Additive mdoel.

the following graph visualises all these components

//insert image//

## 3. Fitting an Arima Model

The ARIMA (AutoRegressive Integrated Moving Average) model has three key parameters, typically written as ARIMA(p, d, q):

1️⃣ AutoRegressive (AR) Term – p
Represents the number of past values (lags) used to predict the current value.

If p = 2, it means the model uses the last two values of the series.


2️⃣ Differencing (I) Term – d
Represents the number of times the series is differenced to make it stationary.

If d = 1, it means the first difference of the series is taken (i.e., X_t - X_{t-1}).

3️⃣ Moving Average (MA) Term – q
Represents the number of lagged forecast errors used to model the series.

If q = 3, it means the model accounts for the last three errors (residuals).

Although there are multiple rigorus statisticals tests and methods to identify theese hyper parameters, we can use the auto_arima method provoded in the pmdarima libraby to get optimal values for these hyper parameters
Moreover we only train out model for the data till September 2021 and trate the data for oct 2021 as the validation or the test set for our model.
On running the auto_arima on th train set we get the following optimal values for our hyper parameters: (p,d,q) = (7,1,0)

The predition of our model for the month of Oct 2021 or the test set and the actual values are plotted together in the following graph

//intert image//

We trate this as our baseline model and compare other models with this baseline model to assess the relative effectiveness of them.

## 4 Fitting Arima Model with Expanding window




